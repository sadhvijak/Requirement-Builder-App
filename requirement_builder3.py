import openai
import json
import re
import os

api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    openai.api_key = api_key
else:
    pass

REQUIREMENT_SKELETON = {
    "main_concept": None,
    "typical_needs": None,
    "stakeholders": None,
    "key_entities": None,
    "high_level_actions": None,
    "success_criteria": None,
    "constraints": None,
    "flow_type": None,
    "trigger_or_entry": None,
    "input_variables": None,
    "queries": None,
    "decision_rules": None,
    "actions": None,
    "output": None
}

EXTRACTION_PROMPT = """
You are a friendly Salesforce Flow consultant having a natural conversation.

The user said: "{user_message}"

What we know so far:
{current_state}

Your tasks:
1. Extract what they clearly mentioned
2. **INTELLIGENTLY INFER** technical details based on their business need:
   - If they want "send email when...", infer flow_type as "Record-Triggered"
   - If they mention "approve", infer key_entities might include approval processes
   - Make reasonable assumptions about stakeholders, constraints, and technical setup
3. ONLY ask questions if there are CRITICAL business details missing:
   - Ask ONLY if the main action is completely unclear
   - Ask ONLY if essential data/fields are missing and cannot be inferred
   - DO NOT ask for minor details or technical specifics
   - If you can make a reasonable assumption, DO IT instead of asking

IMPORTANT: If the requirement has sufficient detail (main concept, action, output are clear), 
DO NOT ask any questions. Just extract the information.

Return your response in this format:

EXTRACTED_INFO:
{{
  "main_concept": "value or null",
  "typical_needs": "value or null",
  "stakeholders": "value or null",
  "key_entities": "value or null",
  "high_level_actions": "value or null",
  "success_criteria": "value or null",
  "constraints": "value or null",
  "flow_type": "value or null",
  "trigger_or_entry": "value or null",
  "input_variables": "value or null",
  "queries": "value or null",
  "decision_rules": "value or null",
  "actions": "value or null",
  "output": "value or null"
}}

QUESTIONS:
[Only list questions if CRITICAL information is missing. If requirement is clear, write "NONE"]
"""

VALIDATION_PROMPT = """
You are a Salesforce Solution Architect reviewing a flow requirement.

Current Requirement State:
{current_state}

Your job: Assess if we have enough BUSINESS INFORMATION to build the flow.

Assessment rules:
1. Check if these CORE business fields are clear:
   - main_concept: What is the flow about?
   - high_level_actions: What should it DO?
   - output: What is the expected result?

2. For technical fields (flow_type, queries, etc.), you can infer reasonable defaults

3. Mark as NEEDS_CLARIFICATION ONLY if:
   - The main purpose of the flow is completely unclear
   - The core ACTION or business logic is missing
   - Essential business context cannot be inferred

4. Be LENIENT - if you can reasonably infer details, mark as OK
5. Require at least 60% completeness before marking as OK (lowered threshold)

Return ONLY valid JSON in this exact format:
{{
  "status": "OK" or "NEEDS_CLARIFICATION",
  "completeness_score": 0-100,
  "issues": ["only list CRITICAL business gaps"],
  "missing_fields": ["only list ESSENTIAL missing fields"],
  "suggestions": ["technical defaults you filled in"],
  "inferred_fields": {{
    "field_name": "your intelligent assumption"
  }}
}}

Be lenient and practical. If enough detail exists to build a basic flow, mark as OK.
"""

FINAL_PROMPT_GENERATOR = """
You are a Salesforce Business Analyst converting collected requirements into a clear, natural language prompt.

Given the requirement details below, write a SIMPLE, CLEAR user requirement in natural language that describes what the flow should do.

Requirements collected:
{filled_schema}

Write the requirement in this style:
"Create a Salesforce [flow_type] flow which [does what]. It should [key actions and logic]. [Any important conditions or rules]."

Example format:
"Create a Salesforce screen flow which takes case number as input and retrieves the corresponding case status and case priority details."

"Create a Salesforce record-triggered flow which sends an email notification to the account owner when an opportunity is marked as closed won and the amount is greater than $100,000."

Guidelines:
- Start with "Create a Salesforce [flow type] flow which..."
- Use simple, clear business language (no technical jargon)
- Describe inputs, actions, conditions, and outputs naturally
- Keep it concise but complete (2-4 sentences max)
- Focus on WHAT it does, not HOW it works technically

Output ONLY the natural language requirement below (no extra formatting):
---
"""


def ask_openai(prompt):
    """Send prompt to OpenAI GPT-4o and return response."""
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful Salesforce Flow consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI: {e}")
        return ""


def extract_json_from_response(text, marker="EXTRACTED_INFO:"):
    """Extract JSON block from LLM response."""
    try:
        if marker and marker in text:
            json_start = text.index(marker) + len(marker)
            end_marker = "QUESTIONS:" if "QUESTIONS:" in text else None
            if end_marker and end_marker in text[json_start:]:
                json_end = text.index(end_marker, json_start)
                json_text = text[json_start:json_end].strip()
            else:
                json_text = text[json_start:].strip()
        else:
            json_text = text.strip()
        
        json_text = json_text.strip()
        start = json_text.find('{')
        end = json_text.rfind('}')
        
        if start >= 0 and end > start:
            json_str = json_text[start:end+1]
            return json.loads(json_str)
            
    except (json.JSONDecodeError, ValueError, AttributeError) as e:
        print(f"Warning: Could not parse JSON - {e}")
        print(f"Problematic text: {text[:200]}...")
    return {}


def extract_questions(text):
    """Extract questions from LLM response."""
    if "QUESTIONS:" in text:
        questions_section = text.split("QUESTIONS:")[1].strip()
        
        # Check if explicitly marked as NONE
        if questions_section.upper().startswith("NONE"):
            return []
        
        # Extract numbered questions
        questions = re.findall(r'\d+\.\s*(.+?)(?=\d+\.|$)', questions_section, re.DOTALL)
        return [q.strip() for q in questions if q.strip() and q.strip().upper() != "NONE"]
    return []


class RequirementBuilder:

    def __init__(self):
        self.state = REQUIREMENT_SKELETON.copy()
        self.conversation_history = []
        self.questions_asked = 0

    def update_state(self, user_input):
        """Process user input, extract info, update state, and ask follow-up questions."""
        self.conversation_history.append(f"User: {user_input}")
        
        prompt = EXTRACTION_PROMPT.format(
            user_message=user_input,
            current_state=json.dumps(self.state, indent=2)
        )
        response = ask_openai(prompt)
        
        extracted_info = extract_json_from_response(response)
        
        for key, value in extracted_info.items():
            if value and value != "null" and key in self.state:
                self.state[key] = value
        
        questions = extract_questions(response)
        
        if questions:
            self.questions_asked += len(questions)
        
        return questions, response

    def validate(self):
        """Validate requirement completeness and logical correctness."""
        prompt = VALIDATION_PROMPT.format(
            current_state=json.dumps(self.state, indent=2)
        )
        validation_response = ask_openai(prompt)
        
        validation_data = extract_json_from_response(validation_response, marker="")
        
        if isinstance(validation_data, dict) and "inferred_fields" in validation_data:
            for key, value in validation_data["inferred_fields"].items():
                if key in self.state and not self.state[key]:
                    self.state[key] = value
                    print(f"   ℹ️  Inferred {key}: {value}")
        
        return validation_data, validation_response

    def is_ready(self, validation_data):
        """Check if requirement is complete enough for generation."""
        if isinstance(validation_data, dict):
            score = validation_data.get("completeness_score", 0)
            status = validation_data.get("status")
            
            # Lowered threshold - 60% and OK status is sufficient
            return status == "OK" and score >= 60
        return False
    
    def ask_for_confirmation(self):
        """Ask user if they want to add anything more."""
        print("\n" + "="*60)
        print("📋 REQUIREMENT SUMMARY")
        print("="*60)
        
        print(f"Flow Type: {self.state.get('flow_type', 'Not specified')}")
        print(f"Main Concept: {self.state.get('main_concept', 'Not specified')}")
        print(f"Actions: {self.state.get('high_level_actions', 'Not specified')}")
        print(f"Output: {self.state.get('output', 'Not specified')}")
        print(f"Success Criteria: {self.state.get('success_criteria', 'Not specified')}")
        print("="*60)
        
        response = input("\n❓ Is there anything more you'd like to add to this requirement? (yes/no): ").strip().lower()
        return response in ['yes', 'y', 'yeah', 'sure', 'yep']

    def generate_final_prompt(self):
        """Generate the final natural language requirement."""
        prompt = FINAL_PROMPT_GENERATOR.format(
            filled_schema=json.dumps(self.state, indent=2)
        )
        return ask_openai(prompt)

    def show_current_state(self):
        """Display current requirement state."""
        print("\n--- Current Requirement State ---")
        for key, value in self.state.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value or '(empty)'}")
        print("-" * 40)


def main():
    builder = RequirementBuilder()
    print("🚀 Salesforce Flow Requirement Builder")
    print("Type 'status' to see current progress, 'quit' to exit.\n")
    
    iteration = 0
    max_iterations = 15
    confirmed = False
    
    while iteration < max_iterations:
        iteration += 1
        
        user_msg = input("\n💬 You: ").strip()
        
        if user_msg.lower() == 'quit':
            print("\nExiting. Goodbye!")
            break
        
        if user_msg.lower() == 'status':
            builder.show_current_state()
            continue
        
        if not user_msg:
            print("Please provide some input.")
            continue
        
        print("\n⏳ Processing...")
        questions, full_response = builder.update_state(user_msg)
        
        if questions:
            print("\n🤖 Bot:")
            for i, q in enumerate(questions, 1):
                print(f"   {i}. {q}")
        else:
            print("\n🤖 Bot: Got it! Let me validate...")
        
        validation_data, validation_text = builder.validate()
        
        if builder.is_ready(validation_data):
            wants_to_add_more = builder.ask_for_confirmation()
            
            if wants_to_add_more:
                print("\n🤖 Bot: Great! Please tell me what else you'd like to add.")
                continue
            else:
                print("\n✅ REQUIREMENT COMPLETE!")
                builder.show_current_state()
                
                print("\n🎯 Generating final requirement...")
                final_prompt = builder.generate_final_prompt()
            
                print("\n" + "="*60)
                print("📝 FINAL USER REQUIREMENT (NLP)")
                print("="*60)
                print(final_prompt)
                print("="*60)
                confirmed = True
                break
        else:
            if isinstance(validation_data, dict):
                score = validation_data.get("completeness_score", 0)
                print(f"\n📊 Completeness: {score}%")
                if validation_data.get("issues"):
                    print(f"⚠️ Issues: {', '.join(validation_data['issues'])}")
                if not questions and score < 60:
                    print("💡 Please provide more details about what the flow should do.")
    
    if iteration >= max_iterations and not confirmed:
        print("\n⚠️ Maximum iterations reached. Please refine your requirements.")


if __name__ == "__main__":
    main()