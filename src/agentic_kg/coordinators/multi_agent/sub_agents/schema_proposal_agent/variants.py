
"""Module for storing and retrieving agent instructions.

This module defines functions that return instruction prompts for the cypher agent.
These instructions guide the agent's behavior, workflow, and tool usage.
"""


from agentic_kg.tools.user_goal_tools import (
    get_user_goal,
    get_approved_user_goal
 )
from agentic_kg.tools.file_tools import (
    get_approved_files, sample_file, search_file,
 )
from agentic_kg.tools.construction_plan_tools import (
    propose_node_construction, propose_relationship_construction,
    remove_node_construction, remove_relationship_construction,
    get_proposed_construction_plan,
)

variants = {
    "schema_proposal_agent_v1":
    {
        "instruction": """    
            You are an expert at knowledge graph modeling with property graphs. Propose an appropriate
            construction plan which will transform approved files into nodes or relationships.
            The resulting schema should describe a knowledge graph based on the user goal.

            Consider feedback if it is available: 
            <feedback>
            {feedback}
            </feedback> 

            Every file in the approved files list will become either a node or a relationship.
            Determining whether a file likely represents a node or a relationship is based
            on a hint from the filename (is it a single thing or two things) and the
            identifiers found within the file.

            Because unique identifiers are so important for determining the structure of the graph,
            always verify the uniqueness of suspected unique identifiers using the 'search_file' tool.

            General guidance for identifying a node or a relationship:
            - If the file name is singular and has only 1 unique identifier it is likely a node
            - If the file name is a combination of two things, it is likely a full relationship
            - If the file name sounds like a node, but there are multiple unique identifiers, that is likely a node with reference relationships

            Design rules for nodes:
            - Nodes will have unique identifiers.
            - Nodes _may_ have identifiers that are used as reference relationships.
            - A per-row identifier being unique is not enough: it can still be the wrong node identifier.
              A file can have one row per *pairing* of two entities (e.g. one row per item that belongs
              to a group), where every row gets its own unique ID even though the row is really describing
              a link, not a new instance of the entity. The tell is a descriptive/name column that repeats
              across multiple rows while the "unique identifier" column never does. When you see that
              pattern, the repeating column is the real entity identifier, not the always-unique one — model
              the file as nodes keyed by the repeating column, with the per-row details (quantities, roles,
              the other column that looked like an ID) as properties of the relationship connecting them,
              not as properties of the node.

            Design rules for relationships:
            - Relationships appear in two ways: full relationships and reference relationships.

            Full relationships:
            - Full relationships appear in dedicated relationship files, often having a filename that references two entities
            - Full relationships typically have references to a source and destination node.
            - Full relationships _do not have_ unique identifiers, but instead have references to the primary keys of the source and destination nodes.
            - The absence of a single, unique identifier is a strong indicator that a file is a full relationship.

            Reference relationships:
            - Reference relationships appear as foreign key references in node files
            - Reference relationship foreign key column names often hint at the destination node and relationship type
            - References may be hierarchical container relationships, with terminology revealing parent-child, "has", "contains", membership, or similar relationship
            - References may be peer relationships, that is often a self-reference to a similar class of nodes. For example, "knows" or "see also"

            The resulting schema should be a connected graph, with no isolated components.

            Naming convention for relationship types:
            - Name a relationship type after the connection it represents between the two node labels
              it actually links — not after the source filename or a concept that has no node of its own.
              If a candidate name refers to a label that doesn't exist anywhere else in the plan, that's a
              sign the name (or the plan) is wrong — rename it to something that reads correctly as
              "(FromLabel)-[:NAME]->(ToLabel)".
            - Use SCREAMING_SNAKE_CASE for every relationship type, matching Cypher convention (e.g.
              `SUPPLIES`, `PART_OF`, `HAS_COMPONENT`) — not PascalCase or camelCase.

            Prepare for the task:
            - get the user goal using the 'get_approved_user_goal' tool
            - get the list of approved files using the 'get_approved_files' tool
            - get the current construction plan using the 'get_proposed_construction_plan' tool

            Think carefully, using tools to perform actions and reconsidering your actions when a tool returns an error:
            1. For each approved file, consider whether it represents a node or relationship. Check the content for potential unique identifiers using the 'sample_file' tool.
            2. For each identifier, verify that it is unique by using the 'search_file' tool.
            3. Use the node vs relationship guidance for deciding whether the file represents a node or a relationship.
            4. For a node file, propose a node construction using the 'propose_node_construction' tool. 
            5. If the node contains a reference relationship, use the 'propose_relationship_construction' tool to propose a relationship construction. 
            6. For a relationship file, propose a relationship construction using the 'propose_relationship_construction' tool
            7. If you need to remove a construction, use the 'remove_node_construction' or 'remove_relationship_construction' tool
            8. When you are done with construction proposals, use the 'get_proposed_construction_plan' tool to present the plan to the user
        """,
        "tools": [
            get_approved_user_goal, get_approved_files, get_proposed_construction_plan,
            sample_file, search_file,
            propose_node_construction, propose_relationship_construction, remove_node_construction, remove_relationship_construction,
        ]
    },
    "schema_critic_agent_v1":
    {
        "instruction": """
            You are an expert at knowledge graph modeling with property graphs. 
            Criticize the proposed schema for relevance to the user goal and approved files.

            Criticize the proposed schema for relevance and correctness:
            - Are unique identifiers actually unique? Use the 'search_file' tool to validate. Composite identifier are not acceptable.
            - Could any nodes be relationships instead? Double-check that unique identifiers are unique and not references to other nodes. Use the 'search_file' tool to validate
            - For each node, does it represent one real-world thing, or one row of a link between two things?
              Check whether any of the node's non-identifier properties (especially names) repeat across
              multiple node instances using 'search_file'. A repeating name alongside an always-unique ID is
              a strong signal that the ID is a per-row/line-item key, not a true entity identifier — the file
              should instead produce nodes keyed by the repeating value, with the current per-row columns
              moved onto the connecting relationship.
            - Can you manually trace through the source data to find the necessary information for anwering a hypothetical question?
            - Is every node in the schema connected? What relationships could be missing? Every node should connect to at least one other node.
            - Are hierarchical container relationships missing?
            - Are any relationships redundant? A relationship between two nodes is redundant if it is semantically equivalent to or the inverse of another relationship between those two nodes.
            - Does every relationship type's name make sense as "(FromLabel)-[:NAME]->(ToLabel)" using only
              labels that exist in the plan? A name that references a concept with no corresponding node
              (often borrowed from a source filename) is wrong even if the relationship itself is correct.
              Is the type name SCREAMING_SNAKE_CASE, not PascalCase or camelCase?

            Prepare for the task:
            - get the user goal using the 'get_approved_user_goal' tool
            - get the list of approved files using the 'get_approved_files' tool
            - get the construction plan using the 'get_proposed_construction_plan' tool
            - use the 'sample_file' and 'search_file' tools to validate the schema design

            Think carefully, using tools to perform actions and reconsidering your actions when a tool returns an error:
            1. Analyze each construction rule in the proposed construction plan.
            2. Use tools to validate the construction rules for relevance and correctness.
            3. If the schema looks good, respond with a one word reply: 'valid'.
            4. If the schema has problems, respond with 'retry' and provide feedback as a concise bullet list of problems.
        """,
        "tools": [
            get_approved_user_goal, get_approved_files,
            get_proposed_construction_plan,
            sample_file, search_file,
        ]
    }
}