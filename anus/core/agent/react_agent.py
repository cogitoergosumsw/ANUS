"""
React Agent module that extends the base agent with reasoning capabilities.
"""

from typing import Dict, List, Any, Optional, Tuple
import json
import logging
import time
import uuid

from anus.core.agent.base_agent import BaseAgent
from anus.models.base.base_model import BaseModel


class ReactAgent(BaseAgent):
    """
    A reasoning agent that follows the React paradigm (Reasoning and Acting).

    This agent implements a thought-action-observation loop for complex reasoning.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        max_iterations: int = 10,
        llm: Optional[BaseModel] = None,
        **kwargs,
    ):
        """
        Initialize a ReactAgent instance.

        Args:
            name: Optional name for the agent.
            max_iterations: Maximum number of thought-action cycles to perform.
            llm: Language model to use for reasoning. If None, a default one will be created.
            **kwargs: Additional configuration options for the agent.
        """
        super().__init__(name=name, **kwargs)
        self.max_iterations = max_iterations
        self.current_iteration = 0

        # Set up language model
        if llm:
            self.llm = llm
        else:
            # Import here to avoid circular imports
            from anus.models.model_router import ModelRouter

            router = ModelRouter()
            self.llm = router.get_default_model()

        # Define React prompts
        self.thought_prompt = """
        You are a reasoning agent solving a complex task. 
        Given the current context and task, generate a thought that would help you make progress.
        
        Task: {task}
        
        Previous steps:
        {history}
        
        Your thought should:
        - Analyze the current situation
        - Consider relevant information
        - Identify what needs to be done next
        - Be detailed and thorough
        
        Thought:
        """

        self.action_prompt = """
        You are a reasoning agent solving a complex task.
        Based on your thought, decide on the next action to take.
        
        Task: {task}
        
        Previous steps:
        {history}
        
        Current thought: {thought}
        
        Available actions:
        - search: Search for information on a topic
        - calculator: Perform mathematical calculations
        - lookup: Look up specific information
        - finish: Complete the task and provide the final answer
        
        Choose an action and provide the input for that action.
        Respond in JSON format like:
        {{
            "action": "action_name",
            "input": {{
                "query": "your query or input"
            }}
        }}
        
        JSON Response:
        """

    def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a task using the React paradigm.

        Args:
            task: The task description to execute.
            **kwargs: Additional parameters for task execution.

        Returns:
            A dictionary containing the execution result and metadata.
        """
        self.update_state(status="executing", task=task)
        self.current_iteration = 0

        # Initialize the context with the task
        context = {"task": task, "thoughts": [], "actions": [], "observations": []}

        # Main React loop
        while self.current_iteration < self.max_iterations:
            try:
                # Generate thought
                thought = self._generate_thought(context)
                context["thoughts"].append(thought)

                # Decide on action
                action_name, action_input = self._decide_action(context)
                action = {"name": action_name, "input": action_input}
                context["actions"].append(action)

                # Check if the action is to finish
                if action_name == "finish":
                    final_answer = action_input.get("answer", "Task completed.")

                    # Log the iteration
                    self.log_action(
                        "iteration",
                        {
                            "iteration": self.current_iteration,
                            "thought": thought,
                            "action": action,
                            "final_answer": final_answer,
                        },
                    )

                    # Build result
                    result = {
                        "task": task,
                        "answer": final_answer,
                        "iterations": self.current_iteration + 1,
                        "context": context,
                    }

                    self.update_state(status="completed")
                    return result

                # Execute action and get observation
                observation = self._execute_action(action_name, action_input)
                context["observations"].append(observation)

                # Log the iteration
                self.log_action(
                    "iteration",
                    {
                        "iteration": self.current_iteration,
                        "thought": thought,
                        "action": action,
                        "observation": observation,
                    },
                )

                # Check if we should terminate
                if self._should_terminate(context):
                    break

                self.current_iteration += 1

            except Exception as e:
                logging.error(f"Error in React execution loop: {str(e)}")
                context["errors"] = context.get("errors", []) + [str(e)]
                break

        # Generate final answer
        final_answer = self._generate_final_answer(context)

        result = {
            "task": task,
            "answer": final_answer,
            "iterations": self.current_iteration + 1,
            "context": context,
        }

        self.update_state(status="completed")
        return result

    def _generate_thought(self, context: Dict[str, Any]) -> str:
        """
        Generate a thought based on the current context.
        
        Args:
            context: The current execution context.
            
        Returns:
            A thought string.
        """
        # Build history from previous steps
        history_text = ""
        for i in range(len(context.get("thoughts", []))):
            history_text += f"Iteration {i+1}:\n"
            history_text += f"Thought: {context['thoughts'][i]}\n"
            
            if i < len(context.get("actions", [])):
                action = context["actions"][i]
                history_text += f"Action: {action['name']} - {json.dumps(action['input'])}\n"
                
            if i < len(context.get("observations", [])):
                history_text += f"Observation: {context['observations'][i]}\n"
                
            history_text += "\n"
        
        # Format the prompt
        prompt = self.thought_prompt.format(
            task=context["task"],
            history=history_text
        )
        
        # Call the language model to generate a thought
        try:
            thought_text = self.llm.generate(prompt, temperature=0.7)
            # Truncate the thought if too long
            if len(thought_text) > 1000:
                thought_text = thought_text[:997] + "..."
            return thought_text
        except Exception as e:
            logging.error(f"Error generating thought: {str(e)}")
            return f"I need to reconsider my approach. Previous error: {str(e)}"

    def _decide_action(self, context: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Decide on the next action to take.

        Args:
            context: The current execution context.

        Returns:
            A tuple of (action_name, action_input).
        """
        # Build history text
        history_text = ""
        for i in range(len(context.get("thoughts", []))):
            history_text += f"Iteration {i+1}:\n"
            history_text += f"Thought: {context['thoughts'][i]}\n"

            if i < len(context.get("actions", [])):
                action = context["actions"][i]
                history_text += (
                    f"Action: {action['name']} - {json.dumps(action['input'])}\n"
                )

            if i < len(context.get("observations", [])):
                history_text += f"Observation: {context['observations'][i]}\n"

            history_text += "\n"

        # Format the prompt
        prompt = self.action_prompt.format(
            task=context["task"],
            history=history_text,
            thought=context["thoughts"][-1] if context["thoughts"] else "",
        )

        # Call the language model to generate an action decision
        try:
            response = self.llm.generate(prompt, temperature=0.2)

            # Parse the JSON response
            try:
                action_decision = json.loads(response)
                action_name = action_decision.get("action", "finish")
                action_input = action_decision.get("input", {})

                if not isinstance(action_input, dict):
                    action_input = {"value": action_input}

                return action_name, action_input

            except json.JSONDecodeError:
                # Handle case where response is not valid JSON
                logging.warning(
                    f"Invalid JSON response for action decision: {response}"
                )

                # Try to extract action name and input using simple heuristics
                if "search" in response.lower():
                    query = response.split("search", 1)[1].strip()
                    return "search", {"query": query}
                elif "calculator" in response.lower():
                    expression = response.split("calculator", 1)[1].strip()
                    return "calculator", {"expression": expression}
                elif "lookup" in response.lower():
                    query = response.split("lookup", 1)[1].strip()
                    return "lookup", {"query": query}
                elif "finish" in response.lower():
                    answer = response.split("finish", 1)[1].strip()
                    return "finish", {"answer": answer}
                else:
                    # Default to finish action
                    return "finish", {"answer": response}

        except Exception as e:
            logging.error(f"Error deciding action: {str(e)}")
            # Default to finish action in case of error
            return "finish", {"answer": f"I encountered an error: {str(e)}"}

    def _execute_action(self, action_name: str, action_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action and return the observation.
        
        Args:
            action_name: The name of the action to execute.
            action_input: The input parameters for the action.
            
        Returns:
            The observation from executing the action.
        """
        # This should be extended to call the appropriate tool
        # For now, implementing basic functionality for common actions
        
        try:
            if action_name == "search":
                # Simulate search action
                query = action_input.get("query", "")
                return {"result": f"Search results for '{query}': [Simulated search result]"}
                
            elif action_name == "calculator":
                # Implement basic calculator functionality
                expression = action_input.get("expression", "")
                try:
                    # Use safe eval to calculate expressions
                    # This should be replaced with a proper calculator tool
                    result = eval(expression, {"__builtins__": {}}, {"abs": abs, "max": max, "min": min, "sum": sum})
                    return {"result": f"Calculator result: {result}"}
                except Exception as calc_error:
                    return {"error": f"Calculator error: {str(calc_error)}"}
                    
            elif action_name == "lookup":
                # Simulate lookup action
                query = action_input.get("query", "")
                return {"result": f"Lookup result for '{query}': [Simulated lookup result]"}
                
            elif action_name == "finish":
                # Just return the answer
                return {"result": f"Final answer: {action_input.get('answer', '')}"}
                
            else:
                # Handle unknown action
                return {"error": f"Unknown action: {action_name}"}
                
        except Exception as e:
            logging.error(f"Error executing action {action_name}: {str(e)}")
            return {"error": f"Error executing {action_name}: {str(e)}"}

    def _should_terminate(self, context: Dict[str, Any]) -> bool:
        """
        Determine if execution should terminate.

        Args:
            context: The current execution context.

        Returns:
            True if execution should terminate, False otherwise.
        """
        # Check if we've reached the maximum iterations
        if self.current_iteration >= self.max_iterations - 1:
            return True

        # Check if the last action was a finish action
        if context.get("actions") and context["actions"][-1].get("name") == "finish":
            return True

        # Check if we've encountered too many errors
        error_count = sum(
            1 for obs in context.get("observations", []) if "error" in obs
        )
        if error_count >= 3:  # Terminate after 3 errors
            return True

        # Additional termination conditions could be added here

        return False

    def _generate_final_answer(self, context: Dict[str, Any]) -> str:
        """
        Generate a final answer based on the context.

        Args:
            context: The current execution context.

        Returns:
            The final answer string.
        """
        # If the last action was a finish, use its answer
        if context.get("actions") and context["actions"][-1].get("name") == "finish":
            return (
                context["actions"][-1].get("input", {}).get("answer", "Task completed.")
            )

        # Otherwise, generate a final answer using the language model
        final_answer_prompt = f"""
        You are a reasoning agent that has been working on a task.
        Based on the thought process and observations, provide a final comprehensive answer.
        
        Task: {context.get('task', '')}
        
        Thought process:
        """

        # Add the thought process to the prompt
        for i in range(len(context.get("thoughts", []))):
            final_answer_prompt += f"\nIteration {i+1}:\n"
            final_answer_prompt += f"Thought: {context['thoughts'][i]}\n"

            if i < len(context.get("actions", [])):
                action = context["actions"][i]
                final_answer_prompt += (
                    f"Action: {action['name']} - {json.dumps(action['input'])}\n"
                )

            if i < len(context.get("observations", [])):
                final_answer_prompt += f"Observation: {context['observations'][i]}\n"

        final_answer_prompt += "\nBased on the above information, provide a concise and accurate final answer to the task."

        try:
            final_answer = self.llm.generate(final_answer_prompt, temperature=0.3)
            return final_answer
        except Exception as e:
            logging.error(f"Error generating final answer: {str(e)}")

            # Fallback to a basic answer
            return "Based on my analysis, I've reached a conclusion, but had difficulty formulating the final answer."
