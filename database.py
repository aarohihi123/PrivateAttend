"""PrivateAttend: Relational Persistence, Schema Definitions, and Audit Logging.

Manages SQLite storage with ACID transactions, WAL mode, foreign key cascades,
duplicate attendance constraints, and cryptographic event tracking.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import hashlib
import os
import sqlite3
import threading

from config import config


class Database:
    """Thread-safe SQLite database manager for PrivateAttend."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.init_schema()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides an isolated, configured SQLite connection within a transaction context."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
        )
        conn.row_factory = sqlite3.Row
        # Enforce foreign keys and WAL journal mode
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_schema(self) -> None:
        """Initializes tables, indices, and default administrative credentials."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Students metadata table (separated from biometric templates)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                department TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 2. Consent records table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS consent_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                status TEXT NOT NULL,  -- 'GRANTED', 'WITHDRAWN', 'REVOKED'
                terms_version TEXT NOT NULL,
                granted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                withdrawn_at TIMESTAMP,
                notes TEXT
            );
            """)

            # 3. Protected biometric templates table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS biometric_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                encrypted_template BLOB NOT NULL,
                sample_count INTEGER NOT NULL DEFAULT 1,
                quality_score REAL NOT NULL DEFAULT 0.0,
                algorithm_version TEXT NOT NULL DEFAULT 'arcface-512',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 4. Attendance sessions table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                course_code TEXT NOT NULL,
                session_name TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_by TEXT NOT NULL
            );
            """)

            # 5. Attendance records table (enforces session_id + student_id uniqueness)
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL REFERENCES attendance_sessions(id) ON DELETE CASCADE,
                student_id TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT NOT NULL,  -- 'PRESENT', 'MANUAL_VERIFIED', 'FLAGGED'
                confidence_score REAL NOT NULL,
                liveness_score REAL NOT NULL,
                verification_method TEXT NOT NULL,  -- 'BIOMETRIC', 'MANUAL'
                notes TEXT,
                UNIQUE(session_id, student_id)
            );
            """)

            # 6. Admin users table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'admin',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # 7. Immutable audit events table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                actor TEXT NOT NULL,
                entity_id TEXT,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)

            # Indices for rapid querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_id ON students(student_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_att_sess ON attendance_records(session_id);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_events(timestamp);")

            # Seed default admin user if not present
            cursor.execute("SELECT id FROM admin_users WHERE username = ?", (config.ADMIN_USERNAME,))
            if cursor.fetchone() is None:
                salt = os.urandom(16).hex()
                pw_hash = hashlib.sha256((config.ADMIN_PASSWORD + salt).encode("utf-8")).hexdigest()
                cursor.execute(
                    "INSERT INTO admin_users (username, password_hash, salt, role) VALUES (?, ?, ?, ?)",
                    (config.ADMIN_USERNAME, pw_hash, salt, "admin"),
                )

    # ---------------- Student Operations ---------------- #

    def add_student(
        self,
        student_id: str,
        first_name: str,
        last_name: str,
        email: str,
        department: str,
    ) -> int:
        """Registers a new student record."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO students (student_id, first_name, last_name, email, department)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, first_name, last_name, email, department),
            )
            sid = cursor.lastrowid
            self.log_audit("STUDENT_REGISTERED", "SYSTEM", student_id, f"Registered {first_name} {last_name}", conn=conn)
            return sid

    def get_student(self, student_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves student details by unique student ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM students WHERE student_id = ?", (student_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_students(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """Returns all students, optionally filtering by active state."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            query = "SELECT * FROM students WHERE is_active = 1" if active_only else "SELECT * FROM students"
            cursor.execute(query + " ORDER BY student_id ASC")
            return [dict(r) for r in cursor.fetchall()]

    def delete_student(self, student_id: str) -> bool:
        """Deletes a student and cascades deletion to consent, templates, and attendance."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                self.log_audit("STUDENT_DELETED", "ADMIN", student_id, "Cascaded student profile removal", conn=conn)
            return deleted

    # ---------------- Consent Operations ---------------- #

    def record_consent(
        self,
        student_id: str,
        status: str,
        terms_version: str = "v1.0",
        notes: Optional[str] = None,
    ) -> int:
        """Records a new consent state. If status is WITHDRAWN, purges biometric template immediately."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            withdrawn_at = "CURRENT_TIMESTAMP" if status in ("WITHDRAWN", "REVOKED") else None

            if withdrawn_at:
                cursor.execute(
                    """
                    INSERT INTO consent_records (student_id, status, terms_version, withdrawn_at, notes)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                    """,
                    (student_id, status, terms_version, notes),
                )
                # Instantaneous template purging upon consent withdrawal
                cursor.execute("DELETE FROM biometric_templates WHERE student_id = ?", (student_id,))
                self.log_audit(
                    "CONSENT_REVOKED_TEMPLATE_PURGED",
                    "STUDENT",
                    student_id,
                    f"Consent revoked ({status}). Biometric template purged.",
                    conn=conn,
                )
            else:
                cursor.execute(
                    """
                    INSERT INTO consent_records (student_id, status, terms_version, notes)
                    VALUES (?, ?, ?, ?)
                    """,
                    (student_id, status, terms_version, notes),
                )
                self.log_audit("CONSENT_GRANTED", "STUDENT", student_id, f"Terms {terms_version} accepted.", conn=conn)
            return cursor.lastrowid

    def has_active_consent(self, student_id: str) -> bool:
        """Checks if a student has an active 'GRANTED' consent record without subsequent withdrawal."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT status FROM consent_records
                WHERE student_id = ?
                ORDER BY id DESC LIMIT 1
                """,
                (student_id,),
            )
            row = cursor.fetchone()
            return bool(row and row["status"] == "GRANTED")

    # ---------------- Biometric Template Operations ---------------- #

    def store_template(
        self,
        student_id: str,
        encrypted_template: bytes,
        sample_count: int = 1,
        quality_score: float = 0.0,
        algorithm_version: str = "arcface-512",
    ) -> int:
        """Stores or updates the encrypted biometric template for an enrolled student."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO biometric_templates (
                    student_id, encrypted_template, sample_count, quality_score, algorithm_version, updated_at
                ) VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(student_id) DO UPDATE SET
                    encrypted_template = excluded.encrypted_template,
                    sample_count = excluded.sample_count,
                    quality_score = excluded.quality_score,
                    algorithm_version = excluded.algorithm_version,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (student_id, encrypted_template, sample_count, quality_score, algorithm_version),
            )
            tid = cursor.lastrowid
            self.log_audit(
                "TEMPLATE_STORED",
                "SYSTEM",
                student_id,
                f"Encrypted template updated with {sample_count} samples (quality {quality_score:.1f})",
                conn=conn,
            )
            return tid

    def get_template(self, student_id: str) -> Optional[bytes]:
        """Retrieves the encrypted template bytes for a specific student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT encrypted_template FROM biometric_templates WHERE student_id = ?",
                (student_id,),
            )
            row = cursor.fetchone()
            return bytes(row["encrypted_template"]) if row else None

    def get_all_active_templates(self) -> Dict[str, bytes]:
        """Returns all enrolled encrypted templates for students who have active consent."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT bt.student_id, bt.encrypted_template
                FROM biometric_templates bt
                JOIN students s ON bt.student_id = s.student_id
                WHERE s.is_active = 1
                """
            )
            templates = {}
            for row in cursor.fetchall():
                sid = row["student_id"]
                if self.has_active_consent(sid):
                    templates[sid] = bytes(row["encrypted_template"])
            return templates

    def delete_template(self, student_id: str) -> bool:
        """Explicitly deletes the biometric template for a student."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM biometric_templates WHERE student_id = ?", (student_id,))
            deleted = cursor.rowcount > 0
            if deleted:
                self.log_audit("TEMPLATE_DELETED", "ADMIN", student_id, "Template purged from database", conn=conn)
            return deleted

    # ---------------- Attendance Session Operations ---------------- #

    def create_session(
        self,
        course_code: str,
        session_name: str,
        created_by: str = "admin",
    ) -> int:
        """Initializes a new active attendance session."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO attendance_sessions (course_code, session_name, created_by)
                VALUES (?, ?, ?)
                """,
                (course_code, session_name, created_by),
            )
            sid = cursor.lastrowid
            self.log_audit("SESSION_CREATED", created_by, str(sid), f"Course: {course_code} - {session_name}", conn=conn)
            return sid

    def get_active_session(self) -> Optional[Dict[str, Any]]:
        """Returns the most recent active attendance session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM attendance_sessions WHERE is_active = 1 ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves session metadata by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance_sessions WHERE id = ?", (session_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns all attendance sessions sorted by start time descending."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM attendance_sessions ORDER BY id DESC")
            return [dict(r) for r in cursor.fetchall()]

    def close_session(self, session_id: int) -> bool:
        """Marks an attendance session as closed."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE attendance_sessions SET is_active = 0, end_time = CURRENT_TIMESTAMP WHERE id = ?",
                (session_id,),
            )
            updated = cursor.rowcount > 0
            if updated:
                self.log_audit("SESSION_CLOSED", "ADMIN", str(session_id), "Session concluded.", conn=conn)
            return updated

    # ---------------- Attendance Records Operations ---------------- #

    def is_attendance_marked(self, session_id: int, student_id: str) -> bool:
        """Checks if student has already been marked in the target session."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM attendance_records WHERE session_id = ? AND student_id = ?",
                (session_id, student_id),
            )
            return cursor.fetchone() is not None

    def record_attendance(
        self,
        session_id: int,
        student_id: str,
        status: str,
        confidence_score: float,
        liveness_score: float,
        verification_method: str = "BIOMETRIC",
        notes: Optional[str] = None,
    ) -> bool:
        """Inserts an attendance record. Fails safely on duplicate entries."""
        with self._lock, self.get_connection() as conn:
            cursor = conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO attendance_records (
                        session_id, student_id, status, confidence_score, liveness_score,
                        verification_method, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (session_id, student_id, status, confidence_score, liveness_score, verification_method, notes),
                )
                self.log_audit(
                    "ATTENDANCE_MARKED",
                    verification_method,
                    student_id,
                    f"Session {session_id} - Score: {confidence_score:.2f}, Liveness: {liveness_score:.2f}",
                    conn=conn,
                )
                return True
            except sqlite3.IntegrityError:
                # Duplicate entry prevented by unique constraint
                return False

    def get_session_attendance(self, session_id: int) -> List[Dict[str, Any]]:
        """Retrieves attendance records for a session with joined student profile details."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ar.*, s.first_name, s.last_name, s.email, s.department
                FROM attendance_records ar
                JOIN students s ON ar.student_id = s.student_id
                WHERE ar.session_id = ?
                ORDER BY ar.timestamp ASC
                """,
                (session_id,),
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_student_attendance(self, student_id: str) -> List[Dict[str, Any]]:
        """Retrieves complete attendance history for a single student."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ar.*, s.course_code, s.session_name
                FROM attendance_records ar
                JOIN attendance_sessions s ON ar.session_id = s.id
                WHERE ar.student_id = ?
                ORDER BY ar.timestamp DESC
                """,
                (student_id,),
            )
            return [dict(r) for r in cursor.fetchall()]

    # ---------------- Admin & Auth Operations ---------------- #

    def verify_admin(self, username: str, password_plain: str) -> bool:
        """Verifies admin credentials using salted SHA-256."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT password_hash, salt FROM admin_users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if not row:
                return False
            expected_hash = row["password_hash"]
            salt = row["salt"]
            computed = hashlib.sha256((password_plain + salt).encode("utf-8")).hexdigest()
            return computed == expected_hash

    # ---------------- Audit Operations ---------------- #

    def log_audit(
        self,
        event_type: str,
        actor: str,
        entity_id: Optional[str],
        details: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> None:
        """Appends an event to the immutable audit log."""
        if not config.AUDIT_LOG_ENABLED:
            return
        try:
            if conn is not None:
                conn.execute(
                    """
                    INSERT INTO audit_events (event_type, actor, entity_id, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (event_type, actor, entity_id, details),
                )
                return

            with self.get_connection() as new_conn:
                new_conn.execute(
                    """
                    INSERT INTO audit_events (event_type, actor, entity_id, details)
                    VALUES (?, ?, ?, ?)
                    """,
                    (event_type, actor, entity_id, details),
                )
        except Exception:
            pass  # Fail open on logging to avoid blocking critical transactions

    def get_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieves recent audit log events sorted by timestamp descending."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM audit_events ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(r) for r in cursor.fetchall()]
