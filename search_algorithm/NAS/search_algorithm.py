import os
import time
import numpy as np
import random
import math
from collections import defaultdict
import sys
import utils
from parallel import ParallelOperater, ParallelConfig
from search_space.search_space_config import SearchSpace


class GraphPASSearch(object):
    """
    Realizing population random initializing and genetic search process with PBG
    """

    def __init__(self, sharing_num, mutation_num, search_space, pbg_strategy="explore"):
        self.sharing_num = sharing_num
        self.mutation_num = mutation_num
        self.search_space = search_space.space_getter()
        self.stack_gcn_architecture = search_space.stack_gcn_architecture
        self.pbg_strategy = pbg_strategy

    def search(self, total_pop, sharing_population, sharing_performance, mutation_selection_probability):
        pbg_mutation_probs = self._calculate_pbg_probability(
            sharing_population,
            strategy=self.pbg_strategy
        )
        print(f"PBG-{self.pbg_strategy} mutation probabilities: {pbg_mutation_probs}")
        # select parents based on wheel strategy.
        parents = self.selection(sharing_population, sharing_performance)
        print("parents:\n", parents)

        # PBG guided mutation
        children = self.pbg_mutation(parents, pbg_mutation_probs, total_pop)
        print("children:\n", children)

        total_pop = total_pop + children
        return children, total_pop

    def _calculate_entropy_corrected(self, population, position):
        mutation_space = self.search_space[self.stack_gcn_architecture[position]]
        n_possible_values = len(mutation_space)
        n_individuals = len(population)
        value_counts = defaultdict(int)
        for individual in population:
            value_counts[individual[position]] += 1

        entropy = 0.0
        for value_index in range(n_possible_values):
            count = value_counts.get(value_index, 0)
            probability = count / n_individuals

            if probability > 0:
                entropy -= probability * math.log(probability)

        return entropy

    def _calculate_max_entropy_corrected(self, position):
        n_possible_values = len(self.search_space[self.stack_gcn_architecture[position]])
        if n_possible_values <= 1:
            return 0.0
        return math.log(n_possible_values)

    def _calculate_pbg_probability(self, population, strategy="explore"):
        if not population:
            return [1.0 / len(self.stack_gcn_architecture)] * len(self.stack_gcn_architecture)

        n_positions = len(self.stack_gcn_architecture)

        mutation_probs = []
        for pos in range(n_positions):
            entropy = self._calculate_entropy_corrected(population, pos)
            max_entropy = self._calculate_max_entropy_corrected(pos)

            if strategy == "exploit":
                if max_entropy > 0:
                    concentration = 1 - (entropy / max_entropy)  
                else:
                    concentration = 1.0
                concentration = max(0.1, min(0.9, concentration))
                mutation_probs.append(concentration)

            else:  # strategy == "explore"
                # PBG-0 Strategy: Biased mutation towards positions with high parameter diversity.
                if max_entropy > 0:
                    normalized_entropy = entropy / max_entropy
                else:
                    normalized_entropy = 0.0
                normalized_entropy = max(0.1, min(0.9, normalized_entropy))
                mutation_probs.append(normalized_entropy)
        total = sum(mutation_probs)
        if total > 0:
            mutation_probs = [p / total for p in mutation_probs]
        else:
            mutation_probs = [1.0 / n_positions] * n_positions

        return mutation_probs

    def selection(self, population, performance):
        print(35 * "=", "select parents based on wheel strategy", 35 * "=")

        fitness = np.array(performance)
        fitness_probility = fitness / sum(fitness)
        fitness_probility = fitness_probility.tolist()

        index_list = [index for index in range(len(fitness))]
        parents = []
        parent_index = np.random.choice(index_list, self.sharing_num, replace=False, p=fitness_probility)
        for index in parent_index:
            parents.append(population[index].copy())
        return parents

    def pbg_mutation(self, parents, pbg_mutation_probs, total_pop):
        print(35 * "=", f"PBG-{self.pbg_strategy} guided mutation", 35 * "=")

        children = []
        for parent in parents:
            child = parent.copy()

            position_to_mutate_list = np.random.choice(
                [gene for gene in range(len(parent))],
                min(self.mutation_num, len(parent)),  
                replace=False,
                p=pbg_mutation_probs
            )

            for mutation_index in position_to_mutate_list:
                mutation_space = self.search_space[self.stack_gcn_architecture[mutation_index]]
                current_val = child[mutation_index]
                possible_vals = [i for i in range(len(mutation_space)) if i != current_val]
                if possible_vals:
                    child[mutation_index] = random.choice(possible_vals)

            attempts = 0
            while child in total_pop and attempts < 100:
                adjust_pos = random.randint(0, len(child) - 1)
                mutation_space = self.search_space[self.stack_gcn_architecture[adjust_pos]]
                current_val = child[adjust_pos]
                possible_vals = [i for i in range(len(mutation_space)) if i != current_val]
                if possible_vals:
                    child[adjust_pos] = random.choice(possible_vals)
                attempts += 1

            if child not in total_pop:
                children.append(child)

        return children

    def updating(self, sharing_children, sharing_children_val_performance_list, sharing_population,
                 sharing_performance):
        print(35 * "=", "updating", 35 * "=")
        print("before sharing_performance:\n", sharing_performance)

        # calculating the average fitness based on top k gnn architecture in sharing population
        _, top_performance = utils.top_population_select(sharing_population,
                                                         sharing_performance,
                                                         top_k=self.sharing_num)
        avg_performance = np.mean(top_performance)

        index = 0
        for performance in sharing_children_val_performance_list:
            if performance > avg_performance:
                sharing_performance.append(performance)
                sharing_population.append(sharing_children[index])
                index += 1
            else:
                index += 1
        print("after sharing_performance:\n", sharing_performance)
        return sharing_population, sharing_performance


class PopulationInitialization(object):
    def __init__(self, initial_num, search_space):
        self.initial_num = initial_num
        self.initial_gnn_architecture_embedding_list = []
        self.initial_gnn_architecture_list = []
        self.search_space = search_space.space_getter()
        self.stack_gcn_architecture = search_space.stack_gcn_architecture

    def initialize_random(self):
        print(35 * "=", "population initializing based on random strategy", 35 * "=")

        while len(self.initial_gnn_architecture_embedding_list) < self.initial_num:
            gnn_architecture_embedding = utils.random_generate_gnn_architecture_embedding(self.search_space,
                                                                                          self.stack_gcn_architecture)
            gnn_architecture = utils.gnn_architecture_embedding_decoder(gnn_architecture_embedding,
                                                                        self.search_space,
                                                                        self.stack_gcn_architecture)
            # gnn architecture genetic embedding based on number
            self.initial_gnn_architecture_embedding_list.append(gnn_architecture_embedding)
            self.initial_gnn_architecture_list.append(gnn_architecture)

        return self.initial_gnn_architecture_embedding_list, self.initial_gnn_architecture_list


class MultiTaskSearch(object):

    def __init__(self, search_parameter, gnn_parameter_dict, search_space):
        self.search_parameter = search_parameter
        self.gnn_parameter_dict = gnn_parameter_dict
        self.search_space = search_space

        # initialize estimators
        self.parallel_estimators = {}
        for task in ['mf', 'bp', 'cc']:
            self.parallel_estimators[task] = ParallelOperater(gnn_parameter_dict[task])

        # reward tracking
        self.q_values = {'mf': 0, 'bp': 0, 'cc': 0}  
        self.p_values = {'mf': 0, 'bp': 0, 'cc': 0}  

        # archive
        self.performance_history = {'mf': [], 'bp': [], 'cc': []}
        self.transfer_history = defaultdict(list)

    def search_operator(self):
        """Multitask Search with PBG"""
        print(35 * "=", "Multitask Search with PBG begin", 35 * "=")

        # initialization
        time_initial = time.time()
        searcher_dict, populations, performances, sharing_populations, sharing_performances = \
            self._initialize_search_components()

        time_initial = time.time() - time_initial
        path = "your path"
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save_initial(path, "multitask_initial_time.txt", time_initial)

        print(35 * "=", "PBG mutation probability will be calculated internally", 35 * "=")

        # Initialize Knowledge Pool and Search History
        knowledge_pools = {'mf': [], 'bp': [], 'cc': []}
        search_history = {'mf': [], 'bp': [], 'cc': []}

        for task in ['mf', 'bp', 'cc']:
            for arch, perf in zip(sharing_populations[task], sharing_performances[task]):
                knowledge_pools[task].append({
                    'architecture': arch,
                    'performance': perf,
                    'task': task,
                    'type': 'initial',
                    'improvement': 0
                })
                search_history[task].append(arch)

        # search parameter
        search_epoch = int(self.search_parameter["search_epoch"])
        alpha = self.search_parameter.get('transfer_alpha', 0.1)

        # time record
        time_search_list = []
        epoch_list = []

        for epoch in range(search_epoch):
            time_search = time.time()

            # record the best performance
            current_bests = {
                task: max(sharing_performances[task]) if sharing_performances[task] else 0
                for task in ['mf', 'bp', 'cc']
            }

            for task in ['mf', 'bp', 'cc']:

                # Calculate Transition Probabilities
                transfer_prob = self._calculate_transfer_probability(
                    task, sharing_populations[task], sharing_performances[task]
                )

                if random.random() < transfer_prob:
                    # Self-Improving Search with PBG
                    self_reward, num_evaluated = self._self_evolution_search(
                        task, searcher_dict[task], populations[task],
                        sharing_populations[task], sharing_performances[task],
                        performances[task], knowledge_pools, search_history
                    )

                    # update self-reward
                    self.q_values[task] = alpha * self.q_values[task] + self_reward

                    for other_task in ['mf', 'bp', 'cc']:
                        if other_task != task:
                            self.p_values[other_task] = alpha * self.p_values[other_task] + self_reward * 0.5

                else:
                    # knowledge transfer with PBG
                    transfer_reward, num_evaluated = self._knowledge_transfer_with_pbg(
                        task, populations[task], sharing_populations[task],
                        sharing_performances[task], performances[task],
                        knowledge_pools, search_history
                    )
                    self.p_values[task] = alpha * self.p_values[task] + transfer_reward
                    for other_task in ['mf', 'bp', 'cc']:
                        if other_task != task:
                            self.q_values[other_task] = alpha * self.q_values[other_task] + transfer_reward * 0.5

            for task in ['mf', 'bp', 'cc']:
                if sharing_performances[task]:
                    self.performance_history[task].append(max(sharing_performances[task]))

            self._save_epoch_results(epoch + 1, sharing_populations, sharing_performances)

            time_search_list.append(time.time() - time_search)
            epoch_list.append(epoch + 1)

        self._print_final_results(sharing_populations, sharing_performances)
        self._analyze_search_performance()

        path = '/logger/graphpas_logger'
        if not os.path.exists(path):
            os.makedirs(path)
        utils.experiment_time_save(path,
                                   "multitask_search_time.txt",
                                   epoch_list,
                                   time_search_list)
    def _calculate_transfer_probability(self, task, population, performance):
        diversity = self._calculate_diversity(population, performance)
        q_value = self.q_values[task]
        p_value = self.p_values[task]

        transfer_prob = 1 / (1 + math.exp(-1 * (diversity + q_value - p_value)))

        return transfer_prob

    def _self_evolution_search(self, task, searchers, population, sharing_population,
                               sharing_performance, performance, knowledge_pools, history):

        # Record the pre-search performance baseline.
        performance_baseline = max(sharing_performance) if sharing_performance else 0

        # calculate mutation probability
        children_embeddings = []
        for searcher in searchers:
            children, population = searcher.search(
                population, sharing_population, sharing_performance, None
            )
            children_embeddings.extend(children)

        children_architectures = [
            utils.gnn_architecture_embedding_decoder(
                embedding, self.search_space.space_getter(), self.search_space.stack_gcn_architecture
            ) for embedding in children_embeddings
        ]

        # parallel estimation
        children_performances = self.parallel_estimators[task].estimation(children_architectures)

        # update
        sharing_population, sharing_performance = searchers[0].updating(
            children_embeddings, children_performances, sharing_population, sharing_performance
        )

        performance.extend(children_performances)
        history[task].extend(children_embeddings)

        reward = 0
        for i, perf in enumerate(children_performances):
            if perf > performance_baseline:
                knowledge_pools[task].append({
                    'architecture': children_embeddings[i],
                    'performance': perf,
                    'task': task,
                    'type': 'self_evolution',
                    'improvement': perf - performance_baseline
                })
                reward += 1

        self.transfer_history[f'self_{task}'].append({
            'epoch': len(self.performance_history[task]),
            'reward': reward,
            'num_children': len(children_embeddings),
            'performance_improvement': max(children_performances + [performance_baseline]) - performance_baseline
        })

        return reward, len(children_architectures)

    def _knowledge_transfer_with_pbg(self, task, population, sharing_population,
                                     sharing_performance, performance, knowledge_pools, history):

        performance_baseline = max(sharing_performance) if sharing_performance else 0

        candidate_architectures = self._select_knowledge_candidates(task, knowledge_pools, history[task])

        if len(candidate_architectures) < 2:
            return 0, 0

        # estimation
        candidate_archs_decoded = [
            utils.gnn_architecture_embedding_decoder(
                arch, self.search_space.space_getter(), self.search_space.stack_gcn_architecture
            ) for arch in candidate_architectures
        ]

        candidate_performances = self.parallel_estimators[task].estimation(candidate_archs_decoded)

        # 3. create searcher
        all_children = []
        all_children_performances = []

        for i, candidate in enumerate(candidate_architectures):
            explore_searcher = GraphPASSearch(
                sharing_num=int(self.search_parameter["sharing_num"]),
                mutation_num=2,
                search_space=self.search_space,
                pbg_strategy="explore"
            )

            exploit_searcher = GraphPASSearch(
                sharing_num=int(self.search_parameter["sharing_num"]),
                mutation_num=2,
                search_space=self.search_space,
                pbg_strategy="exploit"
            )

            explore_child = explore_searcher.pbg_mutation(
                [candidate],
                explore_searcher._calculate_pbg_probability(sharing_population, "explore"),
                history[task] + all_children
            )

            exploit_child = exploit_searcher.pbg_mutation(
                [candidate],
                exploit_searcher._calculate_pbg_probability(sharing_population, "exploit"),
                history[task] + all_children + explore_child
            )

            if explore_child:
                all_children.extend(explore_child)
            if exploit_child:
                all_children.extend(exploit_child)


        if all_children:
            children_architectures = [
                utils.gnn_architecture_embedding_decoder(
                    child, self.search_space.space_getter(), self.search_space.stack_gcn_architecture
                ) for child in all_children
            ]

            children_performances = self.parallel_estimators[task].estimation(children_architectures)
            all_children_performances = children_performances
        else:
            children_performances = []
            all_children_performances = []

        reward = 0
        evaluated_architectures = 0

        for i, (candidate, perf) in enumerate(zip(candidate_architectures, candidate_performances)):
            knowledge_pools[task].append({
                'architecture': candidate,
                'performance': perf,
                'task': task,
                'type': 'knowledge_transfer',
                'improvement': perf - performance_baseline,
                'source': 'other_tasks'
            })

            if perf > performance_baseline:
                population.append(candidate)
                sharing_population.append(candidate)
                sharing_performance.append(perf)
                performance.append(perf)
                history[task].append(candidate)
                reward += 1

            evaluated_architectures += 1

        for i, (child, perf) in enumerate(zip(all_children, all_children_performances)):
            population.append(child)
            sharing_population.append(child)
            sharing_performance.append(perf)
            performance.append(perf)
            history[task].append(child)

            if perf > performance_baseline:
                knowledge_pools[task].append({
                    'architecture': child,
                    'performance': perf,
                    'task': task,
                    'type': 'knowledge_transfer_child',
                    'improvement': perf - performance_baseline,
                    'source': 'other_tasks'
                })
                reward += 1

            evaluated_architectures += 1

        if len(population) > int(self.search_parameter["sharing_num"]) * 3:
            excess = len(population) - int(self.search_parameter["sharing_num"]) * 3
            population = population[excess:]
            sharing_population, sharing_performance = utils.top_population_select(
                population, performance, top_k=self.search_parameter["sharing_num"]
            )

        self.transfer_history[f'transfer_{task}'].append({
            'epoch': len(self.performance_history[task]),
            'reward': reward,
            'num_candidates': len(candidate_architectures),
            'num_children': len(all_children),
            'performance_improvement': max(
                candidate_performances + all_children_performances + [performance_baseline]
            ) - performance_baseline,
            'evaluated_architectures': evaluated_architectures
        })

        return reward, evaluated_architectures

    def _select_knowledge_candidates(self, target_task, knowledge_pools, history, num_candidates=2):
        candidates = []
        source_tasks = [task for task in ['mf', 'bp', 'cc'] if task != target_task]

        for source_task in source_tasks:
            if knowledge_pools[source_task]:
                sorted_knowledge = sorted(
                    knowledge_pools[source_task],
                    key=lambda x: x['performance'],
                    reverse=True
                )

                for item in sorted_knowledge:
                    if item['architecture'] not in history and item['architecture'] not in candidates:
                        candidates.append(item['architecture'])
                        break

        if len(candidates) < num_candidates:
            for source_task in source_tasks:
                if knowledge_pools[source_task]:
                    sorted_knowledge = sorted(
                        knowledge_pools[source_task],
                        key=lambda x: x['performance'],
                        reverse=True
                    )

                    for item in sorted_knowledge:
                        if item['architecture'] not in candidates:
                            candidates.append(item['architecture'])
                            if len(candidates) >= num_candidates:
                                break
                    if len(candidates) >= num_candidates:
                        break

        return candidates[:num_candidates]

    def _ensure_unique_architectures(self, architectures, history):
        unique_architectures = []
        seen_architectures = set()

        for arch in architectures:
            arch_tuple = tuple(arch)
            if arch_tuple not in seen_architectures and arch not in history:
                unique_architectures.append(arch)
                seen_architectures.add(arch_tuple)

        return unique_architectures

    def _calculate_diversity(self, population, performance):
        if len(population) < 2:
            return 0.0

        diversity = 0.0
        unit = 0.001

        for i in range(len(population) - 1):
            for j in range(i + 1, len(population)):
                arch1 = population[i]
                arch2 = population[j]
                diff_count = sum(1 for a, b in zip(arch1, arch2) if a != b)
                diversity += diff_count * unit

        return diversity

    def _analyze_search_performance(self):

        for task in ['mf', 'bp', 'cc']:
            final_performance = max(self.performance_history[task]) if self.performance_history[task] else 0
            initial_performance = self.performance_history[task][0] if self.performance_history[task] else 0
            improvement = final_performance - initial_performance

            self_transfers = self.transfer_history.get(f'self_{task}', [])
            transfer_transfers = self.transfer_history.get(f'transfer_{task}', [])

            total_self_reward = sum(item['reward'] for item in self_transfers)
            total_transfer_reward = sum(item['reward'] for item in transfer_transfers)

    def _initialize_search_components(self):
        searcher_dict = {}
        populations = {'mf': [], 'bp': [], 'cc': []}
        performances = {'mf': [], 'bp': [], 'cc': []}
        sharing_populations = {'mf': [], 'bp': [], 'cc': []}
        sharing_performances = {'mf': [], 'bp': [], 'cc': []}

        for task in ['mf', 'bp', 'cc']:
            searcher_list = []
            parallel_num = int(self.search_parameter["parallel_num"])
            mutation_nums = eval(self.search_parameter["mutation_num"])

            # PBG Strategy Configuration: Alternating Exploration-type and Exploitation-type Searchers
            pbg_strategies = ["exploit", "exploit"] * (parallel_num // 2 + 1)

            for index in range(parallel_num):
                searcher = GraphPASSearch(
                    sharing_num=int(self.search_parameter["sharing_num"]),
                    mutation_num=mutation_nums[index],
                    search_space=self.search_space,
                    pbg_strategy=pbg_strategies[index]
                )
                searcher_list.append(searcher)
            searcher_dict[task] = searcher_list

        for task in ['mf', 'bp', 'cc']:
            print(f"\n{35 * '='} initialize {task} Population {35 * '='}")
            population_initialization = PopulationInitialization(
                int(self.search_parameter["initial_num"]), self.search_space
            )
            initial_embeddings, initial_architectures = population_initialization.initialize_random()

            result = self.parallel_estimators[task].estimation(initial_architectures)

            populations[task] = initial_embeddings
            performances[task] = result

            self.performance_history[task].append(max(result) if result else 0)

            sharing_populations[task], sharing_performances[task] = utils.top_population_select(
                populations[task], performances[task], top_k=self.search_parameter["sharing_num"]
            )

        return searcher_dict, populations, performances, sharing_populations, sharing_performances

    def _save_epoch_results(self, epoch, sharing_populations, sharing_performances):
        path = "./logger/graphpas_logger/"
        if not os.path.exists(path):
            os.makedirs(path)

        for task in ['mf', 'bp', 'cc']:
            utils.experiment_graphpas_data_save(
                path, f"protein_{task}_epoch_{epoch}.txt",
                sharing_populations[task], sharing_performances[task],
                self.search_space.space_getter(), self.search_space.stack_gcn_architecture
            )

    def _print_final_results(self, sharing_populations, sharing_performances):
        print("\n" + 50 * "=" + " the result " + 50 * "=")

        for task in ['mf', 'bp', 'cc']:
            if sharing_performances[task]:
                best_idx = np.argmax(sharing_performances[task])
                best_architecture = sharing_populations[task][best_idx]
                best_performance = sharing_performances[task][best_idx]

                best_architecture_decoded = utils.gnn_architecture_embedding_decoder(
                    best_architecture, self.search_space.space_getter(), self.search_space.stack_gcn_architecture
                )

                print(f"\nTask {task.upper()}:")
                print(f"  The Best Architecture: {best_architecture_decoded}")
                print(f"  The Best performance: {best_performance:.4f}")


if __name__ == "__main__":
    ParallelConfig(True)
    search_space = SearchSpace()

    print("\n" + "=" * 50 + " Multitask search with PBG " + "=" * 50)
    multi_search_parameter = {
        "parallel_num": "4",
        "mutation_num": "[1,1,2,2]",
        "initial_num": "6",
        "sharing_num": "2",
        "search_epoch": "15",
        "transfer_alpha": 0.1
    }

    gnn_parameter_dict = {
        'mf': {"data_cnf": 'mf', "gpu_number": "0,1,2", "epoch_number": "8", "pre_name": "mf_model"},
        'bp': {"data_cnf": 'bp', "gpu_number": "0,1,2", "epoch_number": "8", "pre_name": "bp_model"},
        'cc': {"data_cnf": 'cc', "gpu_number": "0,1,2", "epoch_number": "8", "pre_name": "cc_model"}
    }

    multi_task_searcher = MultiTaskSearch(multi_search_parameter, gnn_parameter_dict, search_space)
    multi_task_searcher.search_operator()