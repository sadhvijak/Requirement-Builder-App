import os
from typing import List, Optional

import streamlit as st
from PIL import Image
import openai
from dotenv import load_dotenv

from requirement_builder2 import RequirementBuilder

load_dotenv()

def _read_api_key() -> Optional[str]:
    key = None
    try:
        if hasattr(st, "secrets") and isinstance(st.secrets, dict) and "OPENAI_API_KEY" in st.secrets:
            key = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    return key

API_KEY = _read_api_key()

if API_KEY:
    openai.api_key = API_KEY

st.set_page_config(page_title="Salesforce Requirement Chat", page_icon="💬", layout="wide")

def get_openai_client():
    """Initialize OpenAI client"""
    try:
        return openai.OpenAI(api_key=API_KEY) if API_KEY else None
    except Exception as e:
        st.error(f"Failed to init OpenAI client: {e}")
        return None


def analyze_images(files: List[st.runtime.uploaded_file_manager.UploadedFile]) -> str:
    """Analyze one or more uploaded images and return a comprehensive extraction of ALL details
    from Salesforce workflow/flow diagrams."""
    if not files:
        return ""

    client = get_openai_client()
    if client is None:
        return ""

    # Prepare image content for OpenAI Vision
    image_contents = []
    for f in files:
        try:
            f.seek(0)
            import base64
            image_data = base64.b64encode(f.read()).decode('utf-8')
            image_contents.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{f.type or 'image/png'};base64,{image_data}",
                    "detail": "high"  # Request high-resolution analysis
                }
            })
        except Exception as e:
            st.error(f"Failed to process image {f.name}: {e}")
            continue

    prompt = """You are an expert Workflow/Process Analyst performing a COMPREHENSIVE analysis of ANY type of workflow, process flow, or diagram.

CRITICAL INSTRUCTIONS: Extract EVERY detail visible in the diagram(s). Do not summarize or skip any information. Adapt your analysis to the type of workflow shown.

STEP 1: IDENTIFY THE WORKFLOW TYPE
First, determine what type of workflow/diagram this is:
- Business Process (BPMN, flowchart, process map)
- Software Workflow (Salesforce Flow, Power Automate, Zapier, n8n, etc.)
- System Architecture/Data Flow
- Decision Tree
- State Machine/State Diagram
- User Journey/Customer Flow
- API/Integration Flow
- DevOps Pipeline/CI/CD
- Other (identify the type)

STEP 2: EXTRACT ALL VISIBLE ELEMENTS SYSTEMATICALLY

Regardless of workflow type, extract ALL of the following:

1. WORKFLOW IDENTITY & METADATA:
   - Workflow name/title
   - Workflow type/category
   - Version (if visible)
   - Creator/owner (if visible)
   - Date/timestamp (if visible)
   - Description or purpose statement
   - Platform/tool used (if identifiable)

2. START/TRIGGER/ENTRY POINTS:
   - What initiates this workflow?
   - Trigger events, conditions, or starting criteria
   - Entry parameters or inputs required
   - Scheduled triggers (time-based, event-based, manual, API call, etc.)
   - Preconditions that must be met

3. INPUT PARAMETERS/VARIABLES/DATA:
   - ALL input parameters with:
     * Parameter/variable name
     * Data type (string, number, object, array, boolean, etc.)
     * Source (user input, API, database, previous step, etc.)
     * Required vs optional
     * Default values
     * Validation rules or constraints

4. PROCESS ELEMENTS/NODES (Extract EVERY element in sequence):
   For EACH visible element/node/step, extract:
   - Element identifier (name, label, ID)
   - Element type (process, decision, action, gateway, subprocess, service call, etc.)
   - Element description or purpose
   - Configuration details visible
   - Connection to other elements (which elements it flows to/from)
   - Sequence order or execution priority

5. DECISION POINTS/GATEWAYS/CONDITIONALS:
   For EACH decision/branch point:
   - Decision name/label
   - Condition type (if-then-else, switch/case, multiple conditions, etc.)
   - ALL branches/outcomes with their EXACT conditions
   - Logical operators (AND, OR, NOT, XOR, etc.)
   - Comparison operators (equals, greater than, contains, matches, etc.)
   - Field/variable names in conditions
   - Values being compared against
   - Default/fallback path

6. ACTIONS/OPERATIONS/TASKS:
   For EACH action/operation:
   - Action type (create, read, update, delete, send, call, transform, validate, etc.)
   - Target system/service/object
   - Operation details (what exactly is being done)
   - Input data/parameters for this action
   - Output data/results produced
   - Error handling for this action

7. DATA OPERATIONS:
   - Database queries (SELECT, INSERT, UPDATE, DELETE with exact criteria)
   - API calls (endpoint, method, headers, body, authentication)
   - Data transformations (mappings, calculations, formatting)
   - Variable assignments (what = what)
   - Data validation rules
   - Field-level mappings (source field → destination field)

8. INTEGRATIONS/EXTERNAL CALLS:
   - External systems called
   - API endpoints
   - Service names
   - Authentication methods
   - Request/response formats
   - Timeout settings
   - Retry logic

9. LOOPS/ITERATIONS/BATCHES:
   - Loop type (for-each, while, do-until, etc.)
   - Collection/data being iterated
   - Loop variable/current item
   - Operations inside the loop
   - Exit conditions
   - Batch size (if applicable)

10. SUBPROCESSES/CHILD WORKFLOWS:
    - Subprocess name/reference
    - When it's invoked
    - Input parameters passed
    - Output values received
    - Synchronous vs asynchronous

11. CALCULATIONS/FORMULAS/EXPRESSIONS:
    - ALL mathematical formulas (extract exact syntax)
    - String manipulations
    - Date/time calculations
    - Logical expressions
    - Functions used
    - Variable references

12. ERROR HANDLING/EXCEPTION PATHS:
    - Try-catch blocks
    - Error conditions handled
    - Fallback actions
    - Retry mechanisms
    - Failure notifications
    - Rollback procedures

13. NOTIFICATIONS/COMMUNICATIONS:
    - Email sending (to whom, subject, body template)
    - SMS/messaging
    - Webhooks
    - Alerts/notifications
    - Approval requests
    - User assignments

14. OUTPUT/RESULTS/END STATES:
    - What is produced/modified by the workflow
    - Output variables/return values
    - Success states
    - Failure states
    - Side effects (records created, files generated, etc.)
    - Final actions before completion

15. TIMING & SCHEDULING:
    - Wait/delay steps
    - Scheduled actions
    - Timeout values
    - SLA/deadline requirements
    - Parallel vs sequential execution

16. PERMISSIONS/SECURITY/ROLES:
    - User roles involved
    - Permission checks
    - Authentication requirements
    - Authorization gates
    - Data access controls

17. VISUAL FLOW STRUCTURE:
    - Complete path from start to all possible end points
    - ALL branches and their conditions
    - Parallel execution paths
    - Merge points
    - Swimlanes (if present) - who does what
    - Color coding meanings (if applicable)

18. ANNOTATIONS/COMMENTS/NOTES:
    - Any descriptive text boxes
    - Comments or explanations
    - Notes about specific steps
    - Warnings or important callouts
    - Version notes or change logs

19. METRICS/MONITORING (if visible):
    - Performance indicators
    - Logging points
    - Monitoring/tracking elements
    - Analytics or reporting outputs

20. SPECIALIZED ELEMENTS (adapt based on diagram type):
    - For BPMN: Pools, lanes, message flows, signals, timers
    - For Software Flows: Connectors, transformers, filters
    - For State Diagrams: States, transitions, guards
    - For Data Flows: Data stores, data sources, transformations
    - Any domain-specific elements unique to this workflow type

FORMAT YOUR RESPONSE AS FOLLOWS:

=== WORKFLOW TYPE & OVERVIEW ===
[Identified workflow type and high-level purpose]

=== WORKFLOW METADATA ===
[Name, version, platform, creation details]

=== TRIGGER/START CONDITIONS ===
[What initiates this workflow and under what conditions]

=== INPUT PARAMETERS ===
[ALL input variables/parameters with complete details]

=== WORKFLOW ELEMENTS (In Execution Order) ===

**[Element 1 ID/Name]** - [Type]
• Purpose: [what it does]
• Configuration: [all visible settings]
• Input: [what goes in]
• Output: [what comes out]
• Conditions: [any conditions for this element]
• Connects to: [next element(s)]
• Notes: [any annotations or special details]

**[Element 2 ID/Name]** - [Type]
• Purpose: [what it does]
• Configuration: [all visible settings]
• Input: [what goes in]
• Output: [what comes out]
• Conditions: [any conditions for this element]
• Connects to: [next element(s)]
• Notes: [any annotations or special details]

[Continue for ALL elements - do not skip any]

=== DECISION POINTS & BRANCHING LOGIC ===
[ALL decision nodes with COMPLETE conditions for each branch]

Decision: [Name]
├─ Branch 1: [Condition] → leads to [Element]
├─ Branch 2: [Condition] → leads to [Element]
└─ Default: [Condition or "Otherwise"] → leads to [Element]

=== DATA OPERATIONS & TRANSFORMATIONS ===
[ALL queries, API calls, calculations, and data manipulations]

=== LOOPS & ITERATIONS ===
[ALL loop structures with entry/exit conditions]

=== ERROR HANDLING & EXCEPTION PATHS ===
[ALL error handling mechanisms and fallback logic]

=== INTEGRATIONS & EXTERNAL SYSTEMS ===
[ALL external service calls, APIs, webhooks]

=== COMPLETE EXECUTION PATHS ===
[Map out ALL possible paths from start to end]

Path 1: Start → [Element A] → [Decision B: if X] → [Element C] → End (Success)
Path 2: Start → [Element A] → [Decision B: if Y] → [Element D] → [Element E] → End (Alternative)
Path 3: Start → [Element A] → [Decision B: else] → Error Handler → End (Failure)
[Continue for all paths]

=== ANNOTATIONS & DOCUMENTATION ===
[ALL visible text, comments, notes, or explanatory content]

=== SWIMLANES/ROLES/RESPONSIBILITIES ===
[If applicable: who/what system handles each part]

=== SPECIALIZED ELEMENTS ===
[Any domain-specific or unique elements not covered above]

CRITICAL RULES FOR EXTRACTION:
✓ Extract EVERY visible element - missing even one element is unacceptable
✓ Include exact text from labels, conditions, and field names
✓ Preserve formulas, expressions, and code exactly as shown
✓ List ALL conditions in decision points (not just "if conditions are met")
✓ Document ALL connections between elements
✓ If text is too small to read: note "Text unclear but element type is [X]"
✓ If multiple diagrams: analyze each separately with clear section headers
✓ If colors have meaning: document the color coding scheme
✓ If icons are used: describe what each icon represents
✓ Capture any legends, keys, or reference information visible
✓ Note any crossed-out or deprecated elements
✓ Identify any incomplete or placeholder elements

Begin your exhaustive analysis now:"""

    try:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    *image_contents
                ]
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=4000,  # Increased for comprehensive extraction
            temperature=0.1   # Lower temperature for more precise extraction
        )
        return response.choices[0].message.content or ""
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
        with st.expander("🖼️ Image Context (Persistent)", expanded=False):
            for idx, summary in enumerate(st.session_state.all_image_summaries, 1):
                st.markdown(f"**Image {idx}:**")
                st.text_area(f"img_{idx}", summary, height=100, disabled=True, label_visibility="collapsed")
    st.divider()
    try:
        src = "not set"
        if API_KEY:
            if hasattr(st, "secrets") and isinstance(st.secrets, dict) and "OPENAI_API_KEY" in st.secrets:
                src = "st.secrets"
            elif os.getenv("OPENAI_API_KEY"):
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
                            reply += "\n\n✅ Your requirement looks complete! You can generate the final requirement below or continue adding modifications."
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
                    st.write(f"⚠️ {i}")
            if vdata.get("missing_fields"):
                st.write("**Missing Fields:**")
                for i in vdata["missing_fields"]:
                    st.write(f"❌ {i}")
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
    if st.button("🚀 Generate Final Requirement", type="primary", use_container_width=True):
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