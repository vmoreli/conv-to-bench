from langgraph.graph import StateGraph, START
from src.graphs.states import BenchmarkBuilderState

from src.graphs.nodes.dataset_nodes import (
    is_programming_related_node,
    extract_instruction_node,
    filter_instructions_node,
    identify_feedback_node,
    generate_checklist_node,
)

from src.graphs.nodes.eval_nodes import (
    eval_solution_node,
)

def create_dataset_generator_graph():
    """Graph responsible for filtering and creating requirements."""
    workflow = StateGraph(BenchmarkBuilderState)
    
    workflow.add_node("is_programming_related", is_programming_related_node)
    workflow.add_node("extract_instruction", extract_instruction_node)
    workflow.add_node("filter_instruction", filter_instructions_node)
    workflow.add_node("identify_feedback", identify_feedback_node)
    workflow.add_node("generate_checklist", generate_checklist_node)
    
    workflow.add_edge(START, "is_programming_related")
    return workflow.compile()

def create_evaluator_graph():
    """Graph responsible for code evaluation against a checklist."""
    workflow = StateGraph(BenchmarkBuilderState)
    
    workflow.add_node("eval_solution", eval_solution_node)
    
    workflow.add_edge(START, "eval_solution")
    return workflow.compile()

# Instantiate apps
dataset_app = create_dataset_generator_graph()
evaluator_app = create_evaluator_graph()