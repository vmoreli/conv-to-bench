is_programming_related_system_prompt = """
You are an expert in conversation analysis. 

Your task is to determine if the following conversation between a user and an AI assistant is programming-related.

**is_programming_related**:
- true if the conversation is related to programming (code requests, debugging, code review, algorithm explanations, language snippets, etc.).
- false otherwise.
"""

is_programming_related_user_prompt = """
Determine if the following conversation is programming-related:

Conversation:

{first_message}
"""

extract_instruction_system_prompt = """
You are an expert in conversation analysis.

Your goal is to identify whether a user made a request involving **writing or modifying code** within a conversation with an AI assistant.

## Task

Analyze the conversation and extract the user's original **coding-related instruction**, following these guidelines:

* Guidelines for **instruction**:
  - Provide a clear, concise, and direct description of the user's original request that involves writing or modifying code.
  - If the user requested to modify an existing code, include the request itself and the exact code snippet that needs to be modified.
  - If the conversation **does not contain** any such coding request, set `"instruction"` to an **empty string** ("").
"""

extract_instruction_user_prompt = """
Analyze the following conversation and extract the user's coding-related instruction.

Conversation:

{conversation}
"""

extract_feedback_system_prompt = """
You are a conversation analysis expert.

Your goal is to analyze dialogues between a user and an AI assistant to identify messages containing implicit or explicit user feedback on the assistant's responses.

Your output must adhere to the `FeedbackSchema`.

---

**Feedback Definitions:**

* **Positive Feedback (+):**
    * Occurs when the user confirms that a suggestion, code, or information provided by the assistant was successful, correct, or met their needs.
    * Includes expressions of gratitude that clearly refer to the usefulness of the previous answer.

* **Negative Feedback (-):**
    * Occurs when the user indicates that the assistant's response was unsatisfactory, incorrect, incomplete, or confusing.
    * Includes requests for repetition or clarification that suggest the previous answer failed.
    * Includes direct corrections made by the user to the assistant's code or information.

---

**Crucial Rules:**

1. **User-Focused:** Only messages with 'role: "user"' can be classified as feedback. The feedback is always from the user *about* the assistant's response.  
2. **Feedback is a Reaction:** A feedback message must be a reaction to one or more previous assistant messages. The first user message in a conversation is never feedback.  
3. **Neutrality is the Default:** Messages that continue the conversation without evaluating the previous answer are **neutral** and must not be listed.  
4. **Silence is NOT Positive:** The absence of a user response is **not** positive feedback. Positive feedback must explicitly acknowledge the usefulness or correctness of the previous answer.  
"""

extract_feedback_user_prompt = """
Analyze the following conversation according to the defined rules and return the feedback messages.

**Conversation:**
{conversation}
"""

generate_checklist_system_prompt = """
You are an expert Quality Assurance (QA) analyst specializing in code verification.

Your expertise lies in translating user requirements and feedback into precise, testable criteria.

---

## Task: Generate an Evaluation Checklist

Your task is to create a checklist of requirements that the code must satisfy, following these steps:

1. Analyze the **'Instruction'** provided by the user.
2. Analyze only the **feedback messages explicitly listed** as feedback sources.
   - Do **not** use any other user messages or context outside this list.
3. Synthesize a checklist based on both sources (Instruction and feedback messages).

If **no feedback messages are listed** (i.e., the list of feedback message IDs is empty),  
then the checklist **must be derived solely from the Instruction**.

---

## Output Format and Rules

* The checklist must consist of **simple, unambiguous Yes/No questions**.
* Each question should test only **one atomic condition**.
* Preface each checklist item with its source:
  * '[I]' — derived from the Instruction.
  * '[Fn]' — derived from feedback message *n* (**where *n* must be one of the message IDs explicitly listed in the feedback list**).
* Ensure all output is formatted as a clear and readable list of checklist items.
"""

generate_checklist_user_prompt = """
Generate the evaluation checklist based on the following inputs:

**Conversation:**
{conversation}

**Instruction:**
{instruction}

**Positive Feedback IDs:**
{positive_feedback_ids}

**Negative Feedback IDs:**
{negative_feedback_ids}
"""

filter_instruction_system_prompt = """
You are a validation system. Your job is to decide if a user's instruction is **valid** or **invalid**.

## Definition
An instruction is **valid** if it's clear and complete enough for an AI to give a reasonable answer.

---

## Guidelines

**Mark as INVALID if:**

1.  **It's missing essential information.**
    * The instruction refers to specific content that isn't there (e.g., "Summarize the following text:" but no text is provided).
2.  **It's too vague or ambiguous.**
    * The instruction uses placeholders (like `[insert_name]`) or is too unclear to understand.

**Mark as VALID if:**

1.  **It's self-contained.**
    * The AI can understand and respond using only the instruction itself.
2.  **It's a general or abstract request.**
    * Instructions like "Explain how SQL works" or "Write a plan to build a website" are **valid** because they don't depend on missing files or previous context.

---

## The Main Test

Ask yourself this: **Could an AI provide a good answer using *only* this instruction?**

* **Yes -> valid**
* **No -> invalid**
"""


filter_instruction_user_prompt = """
User Instruction: {instruction}
"""