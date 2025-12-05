import os
from typing import List, Optional

import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

from requirement_builder import RequirementBuilder

load_dotenv()

def _read_api_key() -> Optional[str]:
    key = None
    try:
        if hasattr(st, "secrets") and isinstance(st.secrets, dict) and "GEMINI_API_KEY" in st.secrets:
            key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("GEMINI_API_KEY")
    return key

API_KEY = _read_api_key()

if API_KEY:
    genai.configure(api_key=API_KEY)

st.set_page_config(page_title="Salesforce Requirement Chat", page_icon="💬", layout="wide")

def get_model(model_name: str = "gemini-2.0-flash"):
    try:
        return genai.GenerativeModel(model_name)
    except Exception as e:
        st.error(f"Failed to init Gemini model: {e}")
        return None


def analyze_images(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Analyze one or more uploaded images and return a concise natural language summary suitable
    to feed into the RequirementBuilder conversation."""
    if not files:
        return ""

    model = get_model("gemini-2.0-flash")
    if model is None:
        return ""

    parts = []
    for f in files:
        try:
            img = Image.open(f)
            parts.append(img)
        except Exception:
            # Fallback to bytes
            f.seek(0)
            parts.append({"mime_type": f.type or "image/png", "data": f.read()})

    prompt = (
        "You are a Salesforce Solution Analyst. The user uploaded Salesforce workflow/flow diagrams. "
        "Carefully read the diagrams and describe the BUSINESS requirement in plain English: goals, key entities, triggers, inputs, actions, decisions, and outputs. "
        "Be concise (5-10 sentences)."
    )

    try:
        resp = model.generate_content(parts + [prompt])
        return resp.text or ""
    except Exception as e:
        st.error(f"Image analysis failed: {e}")
        return ""

if "builder" not in st.session_state:
    st.session_state.builder = RequirementBuilder()
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role: 'user'|'assistant', 'content': str}]
if "last_validation" not in st.session_state:
    st.session_state.last_validation = None
if "image_summary" not in st.session_state:
    st.session_state.image_summary = ""  # PERSISTENT: Stores all image context
if "all_image_summaries" not in st.session_state:
    st.session_state.all_image_summaries = []  # NEW: Store all image analyses
if "pending_images" not in st.session_state:
    st.session_state.pending_images = []
if "draft_requirement" not in st.session_state:
    st.session_state.draft_requirement = ""

builder = st.session_state.builder

st.title("Salesforce Requirement Builder")

with st.sidebar:
    st.subheader("Session")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.image_summary = ""
        st.session_state.all_image_summaries = []
        st.session_state.last_validation = None
        st.session_state.pending_images = []
        st.session_state.builder = RequirementBuilder()  # Reset builder state
        st.rerun()
    
    st.divider()
    
    # NEW: Display persistent image context
    if st.session_state.all_image_summaries:
        with st.expander(" Image Context (Persistent)", expanded=False):
            for idx, summary in enumerate(st.session_state.all_image_summaries, 1):
                st.markdown(f"**Image {idx}:**")
                st.text_area(f"img_{idx}", summary, height=100, disabled=True, label_visibility="collapsed")
    st.divider()
    try:
        src = "not set"
        if API_KEY:
            if hasattr(st, "secrets") and isinstance(st.secrets, dict) and "GEMINI_API_KEY" in st.secrets:
                src = "st.secrets"
            elif os.getenv("GEMINI_API_KEY"):
                src = ".env/env"
        st.caption(f"API Key source: {src}")
    except Exception:
        st.caption("API Key source: unknown")

left, right = st.columns([2, 1])

with left:
    st.subheader("Chat")
    # Display history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # Chat-like composer (supports image attachments)
    with st.container():
        st.markdown("---")
        with st.form("composer", clear_on_submit=True):
            user_msg = st.text_area("Message", placeholder="Describe your requirement or ask about the uploaded workflows…", height=100)
            attachments = st.file_uploader("Attach workflow images (optional)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
            include_images = st.checkbox("Analyze attached images", value=True)
            submitted = st.form_submit_button("Send")

        if submitted and (user_msg or attachments):
            # Start building the composed message
            composed_parts = []
            
            # NEW: Always include previous image context if it exists
            if st.session_state.all_image_summaries:
                combined_image_context = "\n\n".join([
                    f"[Previous Image Context {idx}]:\n{summary}" 
                    for idx, summary in enumerate(st.session_state.all_image_summaries, 1)
                ])
                composed_parts.append(f"=== EXISTING WORKFLOW CONTEXT FROM IMAGES ===\n{combined_image_context}\n")
            
            # Analyze new images if provided
            new_img_summary = ""
            if include_images and attachments:
                with st.spinner("Understanding attached images…"):
                    new_img_summary = analyze_images(attachments)
                    if new_img_summary:
                        # Store this new image analysis
                        st.session_state.all_image_summaries.append(new_img_summary)
                        st.session_state.image_summary = new_img_summary
                        composed_parts.append(f"=== NEW IMAGE ANALYSIS ===\n{new_img_summary}\n")
            
            # Add user's text message
            if user_msg:
                composed_parts.append(f"=== USER REQUEST ===\n{user_msg}")
            
            # Combine all parts
            composed = "\n".join(composed_parts).strip()
            
            # Display only the user's actual message in chat (not the full composed context)
            display_msg = user_msg if user_msg else "[Image uploaded]"
            st.session_state.messages.append({"role": "user", "content": display_msg})
            
            with st.spinner("Thinking…"):
                # Send the FULL context (images + message) to the builder
                qs, _full = builder.update_state(composed)
                vdata, _vfull = builder.validate()
                st.session_state.last_validation = vdata
                
                if qs:
                    reply = "Here are a few questions to clarify:\n" + "\n".join(f"- {q}" for q in qs)
                else:
                    if vdata and isinstance(vdata, dict):
                        status = vdata.get("status")
                        score = vdata.get("completeness_score", 0)
                        issues = vdata.get("issues") or []
                        reply = f"Validation status: {status} (completeness {score}%)."
                        if issues:
                            reply += "\n\n Issues: " + "; ".join(issues)
                        
                        # If requirement is ready, show a hint
                        if status == "OK" and score >= 70:
                            reply += "\n\n Your requirement looks complete! You can generate the final requirement below or continue adding modifications."
                    else:
                        reply = "Processed your input."
                
                st.session_state.messages.append({"role": "assistant", "content": reply})
            st.rerun()

    # Review & Validate (moved under chat)
    st.subheader("Review & Validate")
    with st.expander("Validation Status", expanded=True):
        vdata = st.session_state.last_validation
        if isinstance(vdata, dict):
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Status", vdata.get('status', 'N/A'))
            with col2:
                st.metric("Completeness", f"{vdata.get('completeness_score', 0)}%")
            
            if vdata.get("issues"):
                st.write("**Issues:**")
                for i in vdata["issues"]:
                    st.write(f"   {i}")
            if vdata.get("missing_fields"):
                st.write("**Missing Fields:**")
                for i in vdata["missing_fields"]:
                    st.write(f"  {i}")
        else:
            st.info("Send a message to start validation")

    with st.expander("Draft Requirement (editable)", expanded=False):
        c1, c2 = st.columns([1,1])
        with c1:
            if st.button("Generate Draft from current state", use_container_width=True):
                with st.spinner("Generating draft from current state…"):
                    st.session_state.draft_requirement = builder.generate_final_prompt()
        with c2:
            if st.button("Send Draft to Bot for refinement", use_container_width=True, disabled=not st.session_state.draft_requirement.strip()):
                draft = st.session_state.draft_requirement.strip()
                
                # Include image context with draft refinement
                refinement_context = []
                if st.session_state.all_image_summaries:
                    combined_context = "\n\n".join([
                        f"[Image Context {idx}]:\n{summary}" 
                        for idx, summary in enumerate(st.session_state.all_image_summaries, 1)
                    ])
                    refinement_context.append(f"=== WORKFLOW CONTEXT FROM IMAGES ===\n{combined_context}\n")
                
                refinement_context.append(f"=== DRAFT REQUIREMENT ===\n{draft}\n")
                refinement_context.append("Please refine and confirm this requirement based on our current context.")
                
                full_refinement = "\n".join(refinement_context)
                
                st.session_state.messages.append({"role": "user", "content": "Refining draft requirement..."})
                with st.spinner("Refining draft…"):
                    qs, _ = builder.update_state(full_refinement)
                    vdata, _ = builder.validate()
                    st.session_state.last_validation = vdata
                    reply = "Draft received and refined. "
                    if isinstance(vdata, dict):
                        reply += f"Validation: {vdata.get('status')} ({vdata.get('completeness_score', 0)}%)."
                    if qs:
                        reply += "\n\nClarifications: \n" + "\n".join(f"- {q}" for q in qs)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                st.rerun()
        st.text_area("Draft", key="draft_requirement", height=200)

    st.markdown("---")
    if st.button(" Generate Final Requirement", type="primary", use_container_width=True):
        with st.spinner("Generating final requirement…"):
            final_text = builder.generate_final_prompt()
        st.success("✅ Final requirement generated successfully!")
        st.markdown("### Final Requirement")
        st.code(final_text, language="markdown")
    

with right:
    with st.expander("Current Requirement State", expanded=True):
        state = builder.state
        for k, v in state.items():
            status_icon = "✅" if v else "⭕"
            st.write(f"{status_icon} **{k}**: {v or '_(empty)_'}")