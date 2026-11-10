"""PrivateAttend: Streamlit Administrative & Biometric Console.

Provides 9 interactive management views:
1. Dashboard Overview
2. Live Attendance (Webcam)
3. Image-Based Attendance
4. Student Enrollment (Consent + Multi-Sample)
5. Student Management
6. Attendance History & CSV Export
7. Analytics & Score Distributions
8. Privacy, Consent Registry & Audit Trail
9. System Settings & Threshold Calibration
"""

from typing import Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from enrollment import build_template
from attendance import AttendanceEngine, AttendanceVerificationResult
from config import config
from database import Database
from face_engine import DecisionState, FaceEngine
from liveness import LivenessDetector
from modules.detection import FaceDetector
from modules.preprocessing import assess_image_quality, load_image
from modules.recognition import FaceRecognizer
from privacy import KeyManager, TemplateProtector


# ---------------- Page Setup ---------------- #

st.set_page_config(
    page_title="PrivateAttend | Biometric Attendance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- Global Resource Caching ---------------- #

@st.cache_resource
def get_system_components():
    """Initializes and caches singleton application engines."""
    db = Database()
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    liveness_detector = LivenessDetector()
    protector = TemplateProtector()
    engine = AttendanceEngine(
        db=db,
        detector=detector,
        recognizer=recognizer,
        liveness_detector=liveness_detector,
        protector=protector,
    )
    return db, detector, recognizer, liveness_detector, protector, engine


db, detector, recognizer, liveness_detector, protector, engine = get_system_components()


# ---------------- Sidebar Navigation ---------------- #

st.sidebar.title("🛡️ PrivateAttend")
st.sidebar.caption("Consent-Aware On-Device Biometric System")

nav_choice = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Live Attendance",
        "Image-Based Attendance",
        "Student Enrollment",
        "Student Management",
        "Attendance History",
        "Analytics",
        "Privacy & Consent",
        "System Settings",
    ],
)

st.sidebar.divider()
active_session = db.get_active_session()
if active_session:
    st.sidebar.success(f"🟢 Active Session: {active_session['course_code']}")
    st.sidebar.caption(f"{active_session['session_name']}")
else:
    st.sidebar.warning("🟡 No Active Attendance Session")


# ---------------- 1. Dashboard Overview ---------------- #

if nav_choice == "Dashboard Overview":
    st.title("📊 Attendance Dashboard Overview")
    st.markdown("Real-time summary of student enrollment, active sessions, and attendance metrics.")

    students = db.list_students()
    sessions = db.list_sessions()
    total_students = len(students)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Enrolled Students", total_students)
    with col2:
        active_cnt = 1 if active_session else 0
        st.metric("Active Sessions", active_cnt)
    with col3:
        if active_session:
            records = db.get_session_attendance(active_session["id"])
            st.metric("Present Today", len(records))
        else:
            st.metric("Present Today", 0)
    with col4:
        if active_session and total_students > 0:
            records = db.get_session_attendance(active_session["id"])
            att_rate = (len(records) / total_students) * 100.0
            st.metric("Attendance Rate", f"{att_rate:.1f}%")
        else:
            st.metric("Attendance Rate", "0.0%")

    st.divider()

    # Active Session Control
    st.subheader("Current Attendance Session")
    if active_session:
        st.info(
            f"**Course**: {active_session['course_code']} | "
            f"**Session**: {active_session['session_name']} | "
            f"**Started**: {active_session['start_time']}"
        )
        if st.button("End Current Session", type="primary"):
            db.close_session(active_session["id"])
            st.success("Session closed successfully.")
            st.rerun()
    else:
        with st.expander("➕ Open New Attendance Session", expanded=True):
            with st.form("new_session_form"):
                course_code = st.text_input("Course Code", value="CS-401")
                session_name = st.text_input("Session Title", value="Biometrics & Security Lecture 1")
                submit = st.form_submit_button("Start Session")
                if submit:
                    if course_code and session_name:
                        sid = db.create_session(course_code, session_name)
                        st.success(f"Session {sid} initialized!")
                        st.rerun()
                    else:
                        st.error("Please provide both Course Code and Session Title.")

    # Recent Attendance Activity
    if active_session:
        st.subheader("Live Attendance Feed")
        records = db.get_session_attendance(active_session["id"])
        if records:
            df = pd.DataFrame(records)[
                ["student_id", "first_name", "last_name", "timestamp", "status", "confidence_score", "liveness_score"]
            ]
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No attendance entries recorded yet for this session.")


# ---------------- 2. Live Attendance ---------------- #

elif nav_choice == "Live Attendance":
    st.title("📹 Live Camera Biometric Verification")
    st.markdown("Scan student face for real-time presentation attack detection and open-set recognition.")

    if not active_session:
        st.warning("⚠️ No active attendance session. Please start a session in the Overview tab first.")
    else:
        st.info(f"Target Session: **{active_session['course_code']} - {active_session['session_name']}**")

        camera_photo = st.camera_input("Capture Camera Frame")
        if camera_photo is not None:
            img_bytes = camera_photo.getvalue()
            with st.spinner("Analyzing biometric sample..."):
                result: AttendanceVerificationResult = engine.process_attendance_frame(
                    img_bytes, session_id=active_session["id"]
                )

            col_left, col_right = st.columns([1, 1])

            with col_left:
                st.image(img_bytes, caption="Captured Frame", use_container_width=True)

            with col_right:
                st.subheader("Verification Diagnostics")
                if result.success:
                    st.success(f"✅ **{result.message}**")
                    st.metric("Student ID", result.student_id or "N/A")
                    st.metric("Student Name", result.student_name or "N/A")
                    st.metric("Confidence Score", f"{result.confidence_score:.3f}")
                    st.metric("Liveness Score", f"{result.liveness_score:.3f}")
                else:
                    if result.status_code == "ALREADY_MARKED":
                        st.info(f"ℹ️ {result.message}")
                    elif result.status_code == "SPOOF_REJECTED":
                        st.error(f"🚨 **Presentation Attack Detected**: {result.message}")
                    elif result.status_code == "UNKNOWN_REJECTED":
                        st.warning(f"⛔ **Unknown Subject**: {result.message}")
                    elif result.status_code == "UNCERTAIN_PREDICTION":
                        st.warning(f"⚠️ **Uncertain Match**: {result.message}")
                    else:
                        st.error(f"❌ {result.message}")

                    st.caption(f"Status Code: `{result.status_code}`")
                    st.caption(f"Liveness Probability: `{result.liveness_score:.3f}`")
                    st.caption(f"Matching Confidence: `{result.confidence_score:.3f}`")


# ---------------- 3. Image-Based Attendance ---------------- #

elif nav_choice == "Image-Based Attendance":
    st.title("🖼️ Photo-Based Attendance Verification")
    st.markdown("Upload facial probe photographs to verify attendance for students.")

    sessions = db.list_sessions()
    if not sessions:
        st.warning("No attendance sessions available. Create one first.")
    else:
        sess_map = {f"{s['id']}: {s['course_code']} - {s['session_name']}": s["id"] for s in sessions}
        selected_label = st.selectbox("Select Target Session", list(sess_map.keys()))
        selected_sid = sess_map[selected_label]

        uploaded_file = st.file_uploader("Upload Probe Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            col1, col2 = st.columns(2)
            with col1:
                st.image(file_bytes, caption="Probe Image", use_container_width=True)

            with col2:
                if st.button("Verify & Mark Attendance", type="primary"):
                    with st.spinner("Processing..."):
                        res = engine.process_attendance_frame(
                            file_bytes, session_id=selected_sid
                        )

                    if res.success:
                        st.success(f"✅ {res.message}")
                        st.write(f"**Student**: {res.student_name} ({res.student_id})")
                        st.write(f"**Confidence**: {res.confidence_score:.3f}")
                        st.write(f"**Liveness**: {res.liveness_score:.3f}")
                    else:
                        st.error(f"Result: {res.status_code} — {res.message}")


# ---------------- 4. Student Enrollment ---------------- #

elif nav_choice == "Student Enrollment":
    st.title("📝 Student Enrollment & Consent Registration")
    st.markdown("Register student metadata, record explicit informed consent, and upload facial photos.")

    with st.form("enrollment_form"):
        st.subheader("1. Student Profile Details")
        col_a, col_b = st.columns(2)
        with col_a:
            sid = st.text_input("Student ID (Unique)", placeholder="e.g. STU1001")
            first_name = st.text_input("First Name", placeholder="e.g. Marie")
            email = st.text_input("Email Address", placeholder="e.g. marie.curie@univ.edu")
        with col_b:
            last_name = st.text_input("Last Name", placeholder="e.g. Curie")
            department = st.text_input("Department", placeholder="e.g. Physics")

        st.subheader("2. Informed Biometric Consent (GDPR Art. 9)")
        consent_granted = st.checkbox(
            "I have obtained explicit, informed consent from this student for on-device biometric "
            "attendance verification. The student understands templates are encrypted and may be "
            "revoked at any time."
        )

        st.subheader("3. Facial Training Samples (Multi-Sample Enrollment)")
        sample_files = st.file_uploader(
            "Upload 3 to 5 clear facial photos (different expressions / angles)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
        )

        submit_enrollment = st.form_submit_button("Complete Enrollment", type="primary")

    if submit_enrollment:
        if not sid or not first_name or not last_name or not email or not department:
            st.error("All student metadata fields are required.")
        elif not consent_granted:
            st.error("Biometric enrollment cannot proceed without explicit consent registration.")
        elif not sample_files or len(sample_files) < 1:
            st.error("Please provide at least one facial photograph for enrollment.")
        else:
            # 1. Register student metadata
            res = build_template([sf.read() for sf in sample_files], detector, recognizer)
            if res.template is None:
                st.error("Enrollment aborted. Nothing was saved.")
                for r in res.rejections:
                    st.write(f"- {r}")
            else:
                if not db.get_student(sid):
                    db.add_student(sid, first_name, last_name, email, department)
                db.record_consent(sid, status="GRANTED", terms_version="v1.0")
                db.store_template(sid, protector.encrypt_template(res.template),
                                  sample_count=res.accepted, quality_score=res.mean_quality)
                st.success(f"Enrolled {first_name} {last_name} ({sid}) with {res.accepted} samples.")
                for r in res.rejections:
                    st.warning(r)


# ---------------- 5. Student Management ---------------- #

elif nav_choice == "Student Management":
    st.title("👥 Student Directory & Management")

    students = db.list_students(active_only=False)
    if not students:
        st.info("No students enrolled in the system yet.")
    else:
        search_query = st.text_input("🔍 Search by Student ID, Name, or Department").lower()
        filtered = [
            s for s in students
            if search_query in s["student_id"].lower()
            or search_query in s["first_name"].lower()
            or search_query in s["last_name"].lower()
            or search_query in s["department"].lower()
        ]

        st.dataframe(pd.DataFrame(filtered), use_container_width=True)

        st.divider()
        st.subheader("Manual Attendance Override / Deletion")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("**Manual Attendance Override**")
            with st.form("manual_override_form"):
                sess_list = db.list_sessions()
                s_map = {f"{s['id']}: {s['course_code']}": s["id"] for s in sess_list} if sess_list else {}
                override_sid = st.selectbox("Session", list(s_map.keys())) if s_map else None
                override_stuid = st.selectbox("Student", [s["student_id"] for s in students])
                reason = st.text_input("Override Reason", value="Consent revoked / manual verification")
                submit_override = st.form_submit_button("Record Manual Attendance")
                if submit_override and override_sid:
                    ok = engine.manual_mark_attendance(
                        s_map[override_sid], override_stuid, reason=reason, actor="admin"
                    )
                    if ok:
                        st.success(f"Manual attendance marked for {override_stuid}!")
                    else:
                        st.warning("Could not mark manual attendance (already marked or invalid session).")

        with col_m2:
            st.markdown("**Delete Student Profile & Data**")
            del_student = st.selectbox("Select Student to Delete", [s["student_id"] for s in students])
            if st.button("Permanently Delete Student", type="primary"):
                db.delete_student(del_student)
                st.success(f"Student {del_student} and all biometric data permanently deleted.")
                st.rerun()


# ---------------- 6. Attendance History ---------------- #

elif nav_choice == "Attendance History":
    st.title("📜 Attendance History & Reports")

    sessions = db.list_sessions()
    if not sessions:
        st.info("No attendance sessions found.")
    else:
        sess_dict = {f"{s['id']}: {s['course_code']} - {s['session_name']} ({s['start_time']})": s["id"] for s in sessions}
        selected_session = st.selectbox("Choose Session", list(sess_dict.keys()))
        sid = sess_dict[selected_session]

        records = db.get_session_attendance(sid)
        if records:
            df = pd.DataFrame(records)
            cols = [
                "student_id", "first_name", "last_name", "department", "timestamp",
                "status", "confidence_score", "liveness_score", "verification_method", "notes"
            ]
            st.dataframe(df[cols], use_container_width=True)

            csv_data = engine.export_attendance_csv(sid)
            st.download_button(
                label="📥 Export Attendance as CSV",
                data=csv_data,
                file_name=f"attendance_session_{sid}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No attendance entries recorded for this session.")


# ---------------- 7. Analytics ---------------- #

elif nav_choice == "Analytics":
    st.title("📈 Biometric & System Analytics")
    st.markdown("Visual diagnostics of confidence scores, liveness distributions, and audit activity.")

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT confidence_score, liveness_score, status, verification_method FROM attendance_records")
        rows = cursor.fetchall()

    if not rows:
        st.info("Insufficient attendance data to compute analytics distributions.")
    else:
        df_att = pd.DataFrame([dict(r) for r in rows])

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Recognition Confidence Distribution")
            chart_data = pd.DataFrame({"Confidence": df_att["confidence_score"]})
            st.bar_chart(chart_data)

        with col_b:
            st.subheader("Liveness (PAD) Probability Distribution")
            chart_data2 = pd.DataFrame({"Liveness Score": df_att["liveness_score"]})
            st.bar_chart(chart_data2)

        st.subheader("Verification Breakdown")
        status_counts = df_att["status"].value_counts()
        st.bar_chart(status_counts)


# ---------------- 8. Privacy & Consent ---------------- #

elif nav_choice == "Privacy & Consent":
    st.title("🔒 Privacy, Consent Registry & Audit Trail")
    st.markdown("Auditable GDPR compliance management, instant cryptographic purging, and access logging.")

    tab1, tab2 = st.tabs(["Consent Registry & Revocation", "Security Audit Trail"])

    with tab1:
        st.subheader("Active Consent Status")
        students = db.list_students(active_only=False)
        consent_rows = []
        for s in students:
            has_c = db.has_active_consent(s["student_id"])
            has_tmpl = db.get_template(s["student_id"]) is not None
            consent_rows.append({
                "student_id": s["student_id"],
                "name": f"{s['first_name']} {s['last_name']}",
                "consent_active": "GRANTED" if has_c else "WITHDRAWN",
                "template_stored": "ENCRYPTED" if has_tmpl else "PURGED",
            })

        st.dataframe(pd.DataFrame(consent_rows), use_container_width=True)

        st.divider()
        st.subheader("Revoke Student Consent (Instant Template Purge)")
        st.markdown(
            "Revoking consent immediately deletes the encrypted biometric template from disk and memory. "
            "Subsequent scans will be blocked."
        )
        revoke_sid = st.selectbox("Select Student ID", [s["student_id"] for s in students])
        if st.button("Revoke Consent & Purge Template", type="primary"):
            db.record_consent(revoke_sid, status="WITHDRAWN", notes="User requested consent withdrawal.")
            st.success(f"Consent for {revoke_sid} revoked! Encrypted template permanently purged.")
            st.rerun()

    with tab2:
        st.subheader("Immutable Security Audit Events")
        audit_logs = db.get_audit_logs(limit=100)
        if audit_logs:
            st.dataframe(pd.DataFrame(audit_logs), use_container_width=True)
        else:
            st.caption("No audit events recorded yet.")


# ---------------- 9. System Settings ---------------- #

elif nav_choice == "System Settings":
    st.title("⚙️ System Settings & Threshold Calibration")
    st.markdown("Fine-tune decision boundaries, detector engines, and inspect system operational health.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decision Boundaries (Open-Set)")
        theta_known = st.slider(
            "Known Match Threshold (theta_known)",
            min_value=0.30, max_value=0.90, value=config.THETA_KNOWN, step=0.01,
            help="Minimum cosine similarity required to confirm an enrolled identity."
        )
        theta_unknown = st.slider(
            "Unknown Rejection Threshold (theta_unknown)",
            min_value=0.10, max_value=0.50, value=config.THETA_UNKNOWN, step=0.01,
            help="Similarity below this value triggers definite open-set rejection."
        )
        top2_margin = st.slider(
            "Top-1 vs Top-2 Margin",
            min_value=0.01, max_value=0.20, value=config.TOP2_MARGIN, step=0.01,
            help="Minimum score delta between best and second-best candidate."
        )

        st.subheader("Liveness Detection (PAD)")
        liveness_th = st.slider(
            "Liveness Score Threshold",
            min_value=0.30, max_value=0.90, value=config.LIVENESS_THRESHOLD, step=0.05
        )

    with col2:
        st.subheader("Model Weights & Engine Diagnostics")
        st.write(f"**Detector Backend**: `{detector.backend}`")
        st.write(f"**YuNet Weights**: {'✅ Loaded' if detector.backend == 'yunet' else '⚠️ Haar Fallback Active'}")
        st.write(f"**ArcFace Weights**: {'✅ Loaded' if recognizer.weights_loaded else '⚠️ Mock / Pending Weights'}")
        st.write(f"**LivenessNet Weights**: {'✅ Loaded' if liveness_detector.weights_loaded else '⚠️ Pending Weights'}")
        st.write(f"**Cipher Algorithm**: `{config.CIPHER_ALGORITHM}`")
        st.write(f"**Database**: `{config.DATABASE_PATH}`")

        st.divider()
        st.subheader("Research Dataset Status")
        st.info(
            "**UMDFace Subset (100 IDs x 9 images = 900 total)**: Pending integration. "
            "All dataset-independent architectures and interfaces are operational."
        )
