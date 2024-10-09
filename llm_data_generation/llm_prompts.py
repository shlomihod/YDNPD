# prompt block 1: block for independent variables
def prompt_step_1(schema):
    return (
        f"Consider the following schema: {str(schema)}. \n"
        "You are going to construct a causal graph, relying on your expertise, given only the above schema dictionary defining each variable name and domain/range/categories. When you are unfamiliar with a variable name, infer its identity from the context.\n"
        "You will start by identifying which variable(s) should serve as the root nodes in a directed acyclic graph (DAG), which will represent a structural causal model between all variables (the best root variables are unaffected by any other variables)."
        "Think step by step. Then, provide your final answer (variable names only, EXACTLY as they appear in the schema) within the tags <Answer>...</Answer>, separated by \", \". "
    )

# prompt block 2: block for root vars to next layer
def prompt_step_2():
    return (
        "Now, we are going to relate the root variables to other variables in our causal graph, relying on your expertise. Again, when you are unfamiliar with a variable name, infer its identity from the context.\n"
        "You will now identify relationships between root node variable(s) and remaining variables in the directed acyclic graph. Define a relationship between two variables as 'X -> Y', using the '->' operator to denote a directed edge.\n"
        "Think step by step. Then, provide your final answer (variable names only, with the '->' operator between each directed pair) within the tags <Answer>...</Answer>, separated by ', '."
    )

# prompt block 3: block for any other remaining relationships
def prompt_step_3():
    return (
        "Now, we are going to define any necessary relationships between variables that are NOT root variables in our causal graph, again relying on your expertise.\n"
        "You will now identify relationships between non-root variable(s). Remember, you can define a relationship between two variables as 'X -> Y', using the '->' operator to denote a directed edge.\n"
        "Think step by step. Remember, the graph is a DAG, so be careful not to introduce any cycles! Provide your final answer (variable names only, with the '->' operator between each directed pair) within the tags <Answer>...</Answer>, separated by ', '."
    )

# prompt cycle removal
def prompt_remove_cycles(relationships):
    return (
        f"The list of relationships you defined for our causal graph introduces cycles, making it invalid as a DAG: {str(relationships)} \n\n"
        "Please remove the minimum number of edges to eliminate the cycles, while keeping the most important relationships according to your domain expertise. Make sure, however, that EVERY variable is still included."
        "Return the final list of relationships without any cycles. Provide your final answer (relationships only, with the '->' operator between each directed pair) within the tags <Answer>...</Answer>, separated by ', '."
    )

# prompt block 4: convert to SCM
def prompt_step_4():
    return (
        "In the penultimate step, you are going to specify a set of structural equations, which are functions that relate each node's value to the random variables of its parents.\n"
        "For variables that are root nodes in the causal graph, parameterize a continuous or categorical distribution (depending on the variable type) using your domain expertise from which a random value can be drawn.\n"
        "For variables that have parent nodes, parameterize a conditional distribution, which is a function of the values of the parents of that variable, using your domain expertise, from which a random value can be drawn.\n"
        "Be careful to ensure that the values of the root variables and variables with parents stay within their schema-defined range (in the case of continuous variables), or are valid codes (in the case of categorical variables).\n"
        "Think step by step. Then, provide your final answer as a set of Pyro-like formulas ('X ~ ...', where you insert the formula) within the tags <Answer>...</Answer>, separated by newlines."
    )

# prompt: gen python code one shot
def prompt_generate_pyro_code(pseudocode):
    return f'''
    Finally, you are going to convert the following Pyro-like formulas into executable Pyro code to create a structural causal model (SCM) that I can sample from.
    The formulas specify how each variable is generated based on its parents in a directed acyclic graph (DAG).

    Consider this example:
    <Example>
    X ~ Normal(0, 1)
    Y ~ Normal(2 * X, 1)
    </Example>

    You should convert this into Pyro code like:
    import pyro
    import pyro.distributions as dist

    def model():
        X = pyro.sample("X", dist.Normal(0, 1))
        Y = pyro.sample("Y", dist.Normal(2 * X, 1))
        return {{"X": X, "Y": Y}}

    Remember: module 'pyro.distributions' has no attribute 'TruncatedNormal'.
        
    Be careful to include all functionality INSIDE of the model() function - no helpers! Now, please convert the following Pyro-like formulas into executable Pyro code:\n
    
    {pseudocode}

    Be sure to properly handle any distributions and functional relationships between variables. ALWAYS return Pyro code within the tags <PyroCode>...</PyroCode>.
    '''

# prompt: fix error in pyro code
def prompt_fix_code(pyro_code, error_traceback):
    return f'''
    You generated the following Pyro code, which resulted in an error during execution:
    <PyroCode>
    {pyro_code}
    </PyroCode>

    The error traceback is:
    {error_traceback}

    Please fix only the Pyro code block above, ensuring the corrected code is valid Pyro code and wrapped in <PyroCode>...</PyroCode> tags.'''