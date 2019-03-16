from random import randint, choice
import networkx as nx
EPS_MOVE = "eps"
ACCEPT_SYMBOL = "ACC_SYM"
ACCEPT_STATE = "ACC_STATE"


class State:
    """
    this class represent a single state within an FST machine
    it includes its transitions ( can be weighted )
    init params
    state_name:     the states name ( which is the source )
    transition:     a dictionary of transitions { symbol: target_state, weight<int> }
    is_init:        True if its an initial state
    is_accept:      True if its an accept state
    """
    def __init__(self, state_name, transitions: dict= None, is_init=False, is_accept=False, is_reject=False,
                 artificial_accept=False):
        self._source = state_name
        self._transition = transitions if transitions else {}
        self._is_initial_state = is_init
        self._is_accept_state = is_accept
        self._is_reject_state = is_reject
        self._is_art_accept_state = artificial_accept
        self._weights = ([], 0)
        self._edited = True
        self._weighted = False

    @property
    def id(self):
        return self._source

    @property
    def is_init(self):
        return self._is_initial_state

    @property
    def is_accept(self):
        return self._is_accept_state

    @property
    def is_art_accept(self):
        return self._is_art_accept_state

    @property
    def is_reject(self):
        return self._is_reject_state

    @property
    def source(self):
        return self._source

    @property
    def transitions(self):
        return self._transition

    """
    this function edits a transition according to input 
    """
    def edit_transition(self, symbol, target, weight=1):
        self._edited = True
        # picking by weights is rather expensive, thus a flag will be raise if there are any weights at all
        if weight != 1:
            self._weighted = True
        self._transition[symbol] = (target, weight)

    def _get_acceptable_weights(self):
        # if the machine was edited then update the weights
        if self._edited:
            # reset weights, weights is a list of tuples [ ... ( symbol, target, weight )
            self._weights = [("sym", "<<>>", 0)]
            # the weight of transition i is the difference between i and i-1
            i = 0
            for symbol, (target, weight) in self._transition.items():
                if not target.is_reject:
                    self._weights.append((symbol, target, self._weights[i][2] + weight))
                    i += 1
            # dismiss ("<<>>", 0)
            self._weights = (self._weights[1:], self._weights[-1][2])  # return weights + max weight
            self._edited = False
        return self._weights

    """
        this function randomly picks a transition, according to the weights of the model
    """
    def _rand_acceptable_with_weights(self):
        # get weights as intervals i.e. for symbols=weights <a=2,b=3,c=5>
        # weights is [2,5,10] and max weight is 10
        weights, max_weight = self._get_acceptable_weights()
        # rand a number between  0 <= n <= max - 1
        rand_num = randint(0, max_weight - 1)
        # loop over transitions, and return the first one that bigger then target,
        # if weights is empty return epsilon move
        for symbol, target, weight in weights:
            if weight > rand_num:
                return symbol, target
        return EPS_MOVE, self

    """
        this function randomly picks a transition, with no consideration to the weights of the model
    """
    def _rand_acceptable_without_weights(self):
        # if the transition list is empty return an epsilon move
        tran_list = [(symbol, target) for (symbol, (target, weight)) in self._transition.items()
                     if not target.is_reject]
        if not tran_list:
            return EPS_MOVE, self

        # else, randomly choose a transition
        symbol, target = choice(tran_list)
        return symbol, target

    """
    this function randomly picks a transition
    """
    def _rand_acceptable_transition(self):
        if self._weighted:
            return self._rand_acceptable_with_weights()
        return self._rand_acceptable_without_weights()

    """
    this function returns:
    if a symbol is given -> the next state is returned according the transition function 
    if no symbol is given -> a symbol is raffled according to the weights and then a transition=(symbol, next_state) 
    is returned, the next state cannot be a reject state in this case. 
    """
    def go(self, symbol=None):
        # if there's no transition rule registered for the symbol than state isn't changing
        if symbol is not None:
            return self._transition.get(symbol, self)
        return self._rand_acceptable_transition()


class FST:
    """
    this class represent an FST machine
    init params:
    alphabet:       list of alphabet symbols
    states:         set<preferred>/list/tuple of states<strings/chars/int>
    start_state:    a single initial state
    accept_state:   a single accept state
    transitions:    list of tuples ( source_state, symbol, target_state, weight<int><optional>)
    """
    def __init__(self, alphabet, states, start_state, accept_state, accept_weight, transitions: list):
        self._alphabet = alphabet
        self._states = states
        self._start_state = start_state
        self._accept_state = accept_state
        self._transitions = self._build_transitions(states, start_state, accept_state, accept_weight, transitions)

    def __str__(self):
        out_lines = []
        max_len_state = max(len(max(self._states, key= lambda x: len(x))) + 2, 20)
        max_len_aphabet = max(len(max(self._alphabet, key= lambda x: len(x))) + 2, 20)

        out_lines.append("Alphabet")
        for symbol in self._alphabet:
            out_lines.append(symbol)

        out_lines.append("\n\nStates")
        out_lines.append(" " * int((max_len_state - 6) / 2) + "Source" + " " * int((max_len_state - 6) / 2) + "||"
                         + "     type")

        for state in sorted(self._states):
            if state == ACCEPT_STATE:
                continue
            out_lines.append(state + " " * int(max_len_state - len(state)) + "||"
                             + ("  -initial_state" if self._transitions[state].is_init else "")
                             + ("  -accept_accept" if self._transitions[state].is_accept else "")
                             + ("  -reject_reject" if self._transitions[state].is_reject else "")
                             )

        out_lines.append("\n\nTransitions")
        out_lines.append(" " * int((max_len_state - 6) / 2) + "Source" + " " * int((max_len_state - 6) / 2) + "||" +
                         " " * int((max_len_aphabet - 6) / 2) + "Symbol" + " " * int((max_len_aphabet - 6) / 2) + "||" +
                         " " * int((max_len_state - 6) / 2) + "Target" + " " * int((max_len_state - 6) / 2) + "||" +
                         "     Weight"
                         )

        for state in sorted(self._states):
            if state == ACCEPT_STATE:
                continue
            tran = self._transitions[state].transitions
            for symbol, (target, weight) in tran.items():
                out_lines.append(state + " " * int(max_len_state - len(state)) + "||"
                                 + " " + symbol + " " * int((max_len_aphabet - len(symbol) - 1)) + "||"
                                 + " " + target.id + " " * int((max_len_state - len(target.id) - 1)) + "||"
                                 + " " + str(weight)
                                 )
        return "\n".join(out_lines)

    """
    this function gets for an input a full transition function for the FST and an accept state
    the function return the reject states for the FST
    
     - a graph (gnx) is generated according to the transition 
     - check if path exists from node to accept_state for all nodes in V -> if there isn't then it's a reject_state
    """
    def _get_reject_states(self, transitions, accept_state):
        # build FST graph
        gnx = nx.DiGraph()
        list_edges = []
        for tran in transitions:
            # discard symbols and weights
            source, symbol, target, weight = tran if len(tran) == 4 else list(tran) + [1]
            list_edges.append((source, target))
        gnx.add_edges_from(list_edges)
        # check if there is a path from every node(state) to accept state if not add to reject_states
        reject_states = set()
        for node in gnx:
            if not nx.has_path(gnx, node, accept_state):
                reject_states.add(node)
        return reject_states

    """
    this function builds a transition dictionary:
    { state_name: State_object }
    the state object includes all transitions and weight for a specific state - more info above at State class   
    """
    def _build_transitions(self, states, init_state, accept_state, accept_weight, transitions):
        reject_states = self._get_reject_states(transitions, accept_state)
        # build a dictionary of name to State objects
        state_dict = {q: State(q, is_init=q == init_state, is_accept=q == accept_state,
                               is_reject=q in reject_states) for q in states}
        # add a artificial node for accept state
        state_dict[ACCEPT_STATE] = State(ACCEPT_STATE, artificial_accept=True)
        transitions.append((accept_state, ACCEPT_SYMBOL, ACCEPT_STATE, accept_weight))
        for tran in transitions:
            source, symbol, target, weight = tran if len(tran) == 4 else list(tran) + [1]
            state_dict[source].edit_transition(symbol, state_dict[target], weight=weight)
        return state_dict

    """
    this function runs the machine 
    if a sequence is given -> the function returns the final state and if its accepted by the  machine or not
    if no sequence is given -> the function randomly shuffles according to the weights and returns the sequence
    """
    def go(self, sequence=None):
        # start at initial state
        curr_state = self._transitions[self._start_state]
        # activate states sequentially and return final state
        if sequence is not None:
            for symbol in sequence:
                curr_state = self._transitions[curr_state.id].go(symbol)
                return curr_state, curr_state.is_accept
        else:
            # start from an empty sequence
            sequence = []
            # shuffle symbols until accepted
            while not curr_state.is_art_accept:
                symbol, curr_state = self._transitions[curr_state.id].go()
                sequence.append(symbol)
            return sequence[:-1]


"""
test: the following FST represents the language L={ a^n b^n || n,m > 0 }

                             - a -
                            |     |
                            |     \/
<< q0 start >> --- a --->  << q1 >>
   |                          |
   b                          b
   \/                         \/
<< q3 >> <------- a -----  << q2 >>
|     /\                   |     /\             
|     |                    |     |                   
 - a,b                      - b -               
 
 verse-1 without-weights
      |   a   |   b   |
-----------------------
  q0  |  q1   |  q3   |
  q1  |  q1   |  q2   |
  q2  |  q3   |  q2   |
  q3  |  q3   |  q3   |
  
  
 verse-2 with weights -> weights are such that 'b' sould be longer 
      |   a=weight   |   b=weight   |  acc=weight
-------------------------------------------------
  q0  |     q1=1     |     q3=1     |
  q1  |     q1=3     |     q2=7     |
  q2  |     q3=1     |     q2=7     |  acc=3
  q3  |     q3=1     |     q3=1     |
"""

if __name__ == "__main__":

    # Verse-1
    _alphabet = ["a", "b"]
    _states = {"q0", "q1", "q2", "q3"}
    _init_state = "q0"
    _accept_state = "q2"
    _accept_weight = 1
    _transitions = [
        ("q0", "a", "q1"),
        ("q0", "b", "q3"),
        ("q1", "a", "q1"),
        ("q1", "b", "q2"),
        ("q2", "a", "q3"),
        ("q2", "b", "q2"),
        ("q3", "a", "q3"),
        ("q3", "b", "q3")
    ]
    _fst = FST(_alphabet, _states, _init_state, _accept_state, _accept_weight, _transitions)
    print(_fst)
    _fst.go()

    # Verse-2
    _alphabet = ["a", "b"]
    _states = {"q0", "q1", "q2", "q3"}
    _init_state = "q0"
    _accept_state = "q2"
    _accept_weight = 3
    _transitions = [
        ("q0", "a", "q1"),
        ("q0", "b", "q3"),
        ("q1", "a", "q1", 3),
        ("q1", "b", "q2", 7),
        ("q2", "a", "q3"),
        ("q2", "b", "q2", 7),
        ("q3", "a", "q3"),
        ("q3", "b", "q3")
    ]
    _fst = FST(_alphabet, _states, _init_state, _accept_state, _accept_weight, _transitions)
    print(_fst)
    _fst.go()
    e = 0
