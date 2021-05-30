from typing import List


class MLEEntry:
    def __init__(self, token: str, pos_tag: str):
        self.token = token.lower()
        self.pos_tag = pos_tag

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.token == other.token and self.pos_tag == other.pos_tag

    def __hash__(self):
        return hash(self.token + "/" + self.pos_tag)

    def __repr__(self):
        return f"{self.token} {self.pos_tag}"

    @classmethod
    def from_tag(cls, tag: str):
        return MLEEntry(tag.split("/")[0], tag.split("/")[1])


class CoOccurenceEntry:
    def __init__(self, token1, token2):
        self.token1 = token1
        self.token2 = token2

    # ordered or unordered?
    def __eq__(self, other):
        return isinstance(other, self.__class__) and other.token1 == self.token1 and other.token2 == self.token2

    def __hash__(self):
        # if unordered, modify with sorted
        return hash("___".join([self.token1, self.token2]))

    def __repr__(self):
        return f"{self.token1} {self.token2}"


class TransitionTriplette:
    def __init__(self, co_entry: CoOccurenceEntry, proba):
        self.co_entry = co_entry
        self.proba = proba

    def __lt__(self, other):
        return self.proba < other.proba

    def __le__(self, other):
        return self.proba <= other.proba

    def __eq__(self, other):
        return self.proba == other.proba

    def __ne__(self, other):
        return self.proba != other.proba

    def __gt__(self, other):
        return self.proba > other.proba

    def __ge__(self, other):
        return self.proba >= other.proba

    def __repr__(self):
        return f"{self.co_entry}: {self.proba}"


class State:
    def __init__(self, transition: TransitionTriplette, proba):
        self.transition = transition
        self.state = transition.co_entry.token2
        self.proba = proba

    def get_target(self):
        return self.transition.co_entry.token2

    def __repr__(self):
        return f"State: << {self.state} >>, proba: << {self.proba} >>, transition: << {self.transition} >>"

    def __lt__(self, other):
        return self.proba < other.proba

    def __le__(self, other):
        return self.proba <= other.proba

    def __eq__(self, other):
        return self.proba == other.proba

    def __ne__(self, other):
        return self.proba != other.proba

    def __gt__(self, other):
        return self.proba > other.proba

    def __ge__(self, other):
        return self.proba >= other.proba


class HMM:
    def __init__(self, corpus: List):
        self.word_counts = {}
        self.pos_counts = {}
        self.word_postags = {}
        self.co_postags = {}
        self.corpus = corpus
        self.init_model()

    # to train
    def get_token(self, words=True):
        text = [doc.split(" ") for doc in self.corpus]
        tokenized = []
        for t in text:
            if words:
                tokenized.append([word.split("/")[0] for word in t])
            else:
                tokenized.append([word.split("/")[1] for word in t])
        return tokenized

    def get_bigrams(self, doc, pad=True):
        bigrams = []
        for i in range(0, len(doc) - 1):
            bigrams.append(CoOccurenceEntry(doc[i], doc[i + 1]))
        if pad:
            bigrams.insert(0, CoOccurenceEntry("<s>", doc[0]))
            bigrams.append(CoOccurenceEntry(doc[-1], "<e>"))
        return bigrams

    def add_to_dict(self, entry, dictionary):
        try:
            dictionary[entry] += 1
        except KeyError:
            dictionary[entry] = 1

    def get_from_dict(self, entry, dictionary):
        try:
            return dictionary[entry]
        except KeyError:
            return 0

    def init_unigram_model(self):
        # count all words
        token_corpus = self.get_token(words=True)
        for doc in token_corpus:
            self.add_to_dict("<s>", self.word_counts)
            self.add_to_dict("<e>", self.word_counts)
            for word in doc:
                self.add_to_dict(word, self.word_counts)

        # count all pos tags
        postag_corpus = self.get_token(words=False)
        for doc in postag_corpus:
            # add start and end token
            self.add_to_dict("<s>", self.pos_counts)
            self.add_to_dict("<e>", self.pos_counts)
            for tag in doc:
                self.add_to_dict(tag, self.pos_counts)

    def init_cooccurence_model(self):

        token_corpus = self.get_token(words=False)
        for doc in token_corpus:
            bigrams = self.get_bigrams(doc, pad=True)
            for bigram in bigrams:
                self.add_to_dict(bigram, self.co_postags)

    def init_unigram_pos_model(self):
        tokens = [doc.split(" ") for doc in self.corpus]

        for doc in tokens:
            for token in doc:
                self.add_to_dict(MLEEntry.from_tag(token), self.word_postags)

    def init_model(self):

        # store words
        self.init_unigram_model()
        # store unigrams
        self.init_unigram_pos_model()
        # store bigrams
        self.init_cooccurence_model()

    # helpers

    def p_si_given_sj(self, si, sj):
        c_s1_s2 = self.get_c_of_si_sj(si, sj)
        c_of_s = self.get_c_of_s(sj)
        if c_of_s != 0:
            return c_s1_s2 / c_of_s
        return 0

    def get_c_of_si_sj(self, si, sj):
        try:
            return self.co_postags[CoOccurenceEntry(sj, si)]
        except KeyError:
            return 0

    def get_c_of_s(self, sj):
        try:
            return self.pos_counts[sj]
        except KeyError:
            return 0

    def p_wk_given_si(self, wk, si):
        c_wk_s = self.get_c_wk_s(wk, si)
        c_of_si = self.get_c_of_s(si)
        if c_of_si != 0:
            return c_wk_s / c_of_si
        return 0

    def get_c_wk_s(self, wk, si):
        try:
            return self.word_postags[MLEEntry(wk, si)]
        except KeyError:
            return 0

    def prune_states(self, states):
        best_transitions = {}
        for state in states:
            try:
                if state.transition.proba > best_transitions[state.state].transition.proba:
                    best_transitions[state.state] = state
            except KeyError:
                # means for this state there is no current best
                best_transitions[state.state] = state
        # return best states
        return best_transitions.values()

    def get_emission_and_prune(self, transitions, word):
        new_states = []
        for transition in transitions:
            emission_proba = self.p_wk_given_si(word, transition.co_entry.token2) * transition.proba
            new_state = State(transition=transition, proba=emission_proba)
            new_states.append(new_state)

        # prune
        return [state for state in new_states if state.proba > 0]

    def tag(self, text: List[str]):
        # ["some", "entries", "here"]

        # {timestep: [State1, State2, ...]}
        timesteps = {}

        # calc from start to first tag
        timesteps[0] = self.init_states(text[0])

        # for each word in text
        for i in range(1, len(text)):
            timesteps[i] = self.get_states_for_timestep(timesteps[i - 1], text[i])

        # finally for State --> <e>
        timesteps[len(text)] = self.final_state(timesteps[len(text)-1])

        return timesteps

    def init_states(self, word):
        curr_tag = "<s>"
        candidates = {}
        # calc each possible transition, prune 0 probas
        for next_tag in self.pos_counts.keys():
            # get transition probas
            proba = self.p_si_given_sj(next_tag, curr_tag)
            if proba > 0:
                transition = TransitionTriplette(CoOccurenceEntry(curr_tag, next_tag), proba)
                try:
                    candidates[next_tag].append(transition)
                except KeyError:
                    candidates[next_tag] = [transition]

        # select max for each next_state
        max_transition_per_next_step = []
        for key in candidates.keys():
            max_transition_per_next_step.append(max(candidates[key]))

        # create new states with max transitions
        outcome = self.get_emission_and_prune(max_transition_per_next_step, word)
        print(f"word: {word}")
        print(outcome)
        print("\n")
        return outcome

    def get_states_for_timestep(self, states: List[State], word: str):

        candidates = {}
        # for every state at this timestep, calc all possible steps
        for state in states:
            curr_tag = state.state

            # calc the proba to get into next step
            for next_tag in self.pos_counts.keys():
                proba = self.p_si_given_sj(next_tag, curr_tag)
                if proba > 0:
                    new_proba = proba*state.proba
                    # if transition proba, add to candidates for next_steü
                    transition = TransitionTriplette(CoOccurenceEntry(curr_tag, next_tag), new_proba)
                    try:
                        candidates[next_tag].append(transition)
                    except KeyError:
                        candidates[next_tag] = [transition]

        # select max for each next_state
        max_transition_per_next_step = []
        for key in candidates.keys():
            max_transition_per_next_step.append(max(candidates[key]))

        # create new states with max transitions
        outcome = self.get_emission_and_prune(max_transition_per_next_step, word)
        print(f"Word: {word}")
        print(outcome)
        print("\n")
        return outcome

    def final_state(self, states: List[State]):
        next_tag = "<e>"

        candidates = []
        for state in states:
            curr_tag = state.state
            proba = self.p_si_given_sj(next_tag, curr_tag)
            if proba > 0:
                transition = TransitionTriplette(CoOccurenceEntry(curr_tag, next_tag), proba)
                new_proba = state.proba * proba
                candidates.append(State(transition, new_proba))
        best = self.get_best_candidate(candidates)
        print("Final transition:")
        print(best)
        return best

    def get_best_candidate(self, candidates):
        best = 0
        best_candidate = None

        for can in candidates:
            if can.proba > best:
                best_candidate = can
        return best_candidate


corpus = ["the/D cat/N can/VA fish/VV a/D trout/N in/P seconds/N",
          "the/D fish/N swim/VV in/P the/D can/N",
          "workers/N can/VV the/D food/N"]

mlemodel = HMM(corpus)

to_tag = "the workers can can the food in a can"
to_tag = to_tag.split(" ")
probabilites = mlemodel.tag(to_tag)

#print("\n\n\n")
#print(probabilites)


