#include "optimizer.hpp"

bool Optimizer::dispatch(Message const &message, unsigned int id) {
    bool global_update = false;
    switch (message.code) {
        case Message::exploration_message: {
            // A message travelling downward in the dependency graph
            Bitmask const &parent = message.sender_tile;                // The points captured
            Bitmask const &capture_set = message.recipient_capture;  // The points captured
            Bitmask const &feature_set = message.recipient_feature;  // The features (before pruning)
            bool is_root = capture_set.count() == capture_set.size();
            Task task(capture_set, feature_set, m_dataset,
                      m_local_states[id].column_buffer);  // A vertex to represent the problem
            task.scope(message.scope);
            task.create_children(m_dataset, m_local_states[id].neighbourhood, m_local_states[id].column_buffer,
                                 id);  // Populate the thread's local cache with child instances
            translation_type order;

            vertex_accessor vertex;
            bool inserted = store_self(task.capture_set(), task, vertex);
            // if this subproblem already existed, we still have to update the scope
            if (!inserted) {
                vertex -> second.scope(message.scope);
            }

            store_children(vertex->second, id);

            if (is_root) {  // Update the optimizer state
                float root_upperbound = 1.0;
                if (m_config.upperbound_guess > 0.0) {
                    root_upperbound = std::min(root_upperbound, m_config.upperbound_guess);
                }
                vertex->second.update(m_config, vertex->second.lowerbound(), root_upperbound, -1);
                this->root = vertex->second.capture_set();
                this->translator = vertex->second.order();
                global_update = update_root(vertex->second.lowerbound(), vertex->second.upperbound());
            } else {  // Connect and signal parents
                adjacency_accessor parents;
                link_to_parent(parent, message.features, message.signs, message.scope, vertex->second.capture_set(),
                               vertex->second.order(), parents);
                signal_exploiters(parents, vertex->second, id);
            }

            if (m_config.reference_LB || message.scope >= vertex->second.upperscope()) {
                send_explorers(vertex->second, message.scope, id);
            }

            break;
        }
        case Message::exploitation_message: {
            Bitmask const &identifier = message.recipient_tile;
            vertex_accessor vertex, left, right;

            load_self(identifier, vertex);

            if (vertex->second.uncertainty() == 0 ||
                (!m_config.reference_LB &&
                 vertex->second.lowerbound() >= vertex->second.upperscope() - std::numeric_limits<float>::epsilon())) {
                break;
            }

            bool update = load_children(vertex->second, message.features, id);

            bool is_root = vertex->second.capture_set().count() == vertex->second.capture_set().size();
            if (is_root) {  // Update the optimizer state
                global_update = update_root(vertex->second.lowerbound(), vertex->second.upperbound());
            } else {
                adjacency_accessor parents;  // find backward look-up entry
                load_parents(identifier, parents);
                signal_exploiters(parents, vertex->second, id);  // Signal parents
            }

            break;
        }
        default: {
            std::stringstream reason;
            reason << "Unsupported Message Type: " << message.code;
            throw IntegrityViolation("Optimizer::dispatch", reason.str());
        }
    }
    return global_update;
}

bool Optimizer::load_children(Task &task, Bitmask const &signals, unsigned int id) {
    float lower = task.base_objective(), upper = task.base_objective();
    int optimal_feature = -1;
    bound_accessor bounds;
    m_graph.bounds.find(bounds, task.capture_set());
    for (bound_iterator iterator = bounds->second.begin(); iterator != bounds->second.end(); ++iterator) {
        int feature = std::get<0>(*iterator);

        if (signals.get(feature)) {  // An update is pending
            bool ready = true;
            for (int k = 0; k < 2; ++k) {
                vertex_accessor child;
                child_accessor key;
                ready =
                    ready &&
                    m_graph.children.find(key, std::make_pair(task.capture_set(), k ? -(feature + 1) : (feature + 1))) &&
                    m_graph.vertices.find(child, key->second);
                if (ready) {
                    m_local_states[id].neighbourhood[2 * feature + k] = child->second;
                }
            }

            if (ready) {
                float split_lower, split_upper;
                Task const &left = m_local_states[id].neighbourhood[2 * feature];
                Task const &right = m_local_states[id].neighbourhood[2 * feature + 1];

                if (m_config.rule_list) {
                    float lower_negative = left.lowerbound() + right.base_objective();
                    float lower_positive = left.base_objective() + right.lowerbound();
                    split_lower = std::min(lower_negative, lower_positive);
                    float upper_negative = left.upperbound() + right.base_objective();
                    float upper_positive = left.base_objective() + right.upperbound();
                    split_upper = std::min(upper_negative, upper_positive);
                } else {
                    split_lower = left.lowerbound() + right.lowerbound();
                    split_upper = left.upperbound() + right.upperbound();
                }

                std::get<1>(*iterator) = split_lower;
                std::get<2>(*iterator) = split_upper;
            }
        }

        if (m_config.similar_support) {
            if (iterator != bounds->second.begin()) {  // Comparison with previous feature
                unsigned int i, j;
                float j_lower, j_upper;
                i = std::get<0>(*iterator);
                --iterator;
                j = std::get<0>(*iterator);
                j_lower = std::get<1>(*iterator);
                j_upper = std::get<2>(*iterator);
                ++iterator;

                float distance = m_dataset.distance(task.capture_set(), i, j, m_local_states[id].column_buffer);
                std::get<1>(*iterator) = std::max(std::get<1>(*iterator), j_lower - distance);
                std::get<2>(*iterator) = std::min(std::get<2>(*iterator), j_upper + distance);
            }

            {  // Comparison with next feature
                unsigned int i, j;
                float j_lower, j_upper;
                i = std::get<0>(*iterator);
                ++iterator;
                if (iterator != bounds->second.end()) {
                    j = std::get<0>(*iterator);
                    j_lower = std::get<1>(*iterator);
                    j_upper = std::get<2>(*iterator);
                    --iterator;

                    float distance = m_dataset.distance(task.capture_set(), i, j, m_local_states[id].column_buffer);
                    std::get<1>(*iterator) = std::max(std::get<1>(*iterator), j_lower - distance);
                    std::get<2>(*iterator) = std::min(std::get<2>(*iterator), j_upper + distance);
                } else {
                    --iterator;
                }
            }
        }

        if (std::get<1>(*iterator) > task.upperscope()) {
            continue;
        }
        if (std::get<2>(*iterator) < upper) {
            optimal_feature = std::get<0>(*iterator);
        }
        lower = std::min(lower, std::get<1>(*iterator));
        upper = std::min(upper, std::get<2>(*iterator));
    }
    if (lower > task.upperscope()) { 
        return task.update(m_config, task.upperscope(), upper, optimal_feature); 
    } // ^ if lower is > upperscope, then lower may not be a true lower bound b/c of line 160.
      // (In this case, base objective and all splits are all above upper scope, 
      // but we do not know the exact lower bound)
    return task.update(m_config, lower, upper, optimal_feature);
}

bool Optimizer::load_parents(Bitmask const &identifier, adjacency_accessor &parents) {
    return m_graph.edges.find(parents, identifier);
}

bool Optimizer::load_self(Bitmask const &identifier, vertex_accessor &self) {
    return m_graph.vertices.find(self, identifier);
}

bool Optimizer::store_self(Bitmask const &identifier, Task const &value, vertex_accessor &self) {
    return m_graph.vertices.insert(self, std::make_pair(identifier, value));
}

void Optimizer::store_children(Task &task, unsigned int id) {
    bound_accessor bounds;
    bool inserted = m_graph.bounds.insert(bounds, task.capture_set());
    int optimal_feature = -1;
    float lower = task.base_objective(), upper = task.base_objective();
    Bitmask const &features = task.feature_set();
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            if (m_config.feature_transform == false) {
                for (int sign = -1; sign <= 1; sign += 2) {
                    key_type child_key(m_local_states[id].neighbourhood[2 * j + (sign < 0 ? 0 : 1)].capture_set(), 0);
                    vertex_accessor child;
                    if (m_graph.vertices.find(child, child_key)) {
                        m_local_states[id].neighbourhood[2 * j + (sign < 0 ? 0 : 1)] = child->second;
                    }
                }
            }

            Task &left = m_local_states[id].neighbourhood[2 * j];
            Task &right = m_local_states[id].neighbourhood[2 * j + 1];

            float split_lower, split_upper;
            if (m_config.rule_list) {
                float lower_negative = left.lowerbound() + right.base_objective();
                float lower_positive = left.base_objective() + right.lowerbound();
                split_lower = std::min(lower_negative, lower_positive);
                float upper_negative = left.upperbound() + right.base_objective();
                float upper_positive = left.base_objective() + right.upperbound();
                split_upper = std::min(upper_negative, upper_positive);
            } else {
                split_lower = left.lowerbound() + right.lowerbound();
                split_upper = left.upperbound() + right.upperbound();
            }
            if (inserted) {
                // If we're visiting children for the first time in our graph, store bounds
                bounds->second.push_back(std::tuple<int, float, float>(j, split_lower, split_upper));
            }
            if (split_lower > task.upperscope()) {
                continue;
            }
            if (split_upper < upper) {
                optimal_feature = j;
            }
            lower = std::min(lower, split_lower);
            upper = std::min(upper, split_upper);
        }
    }
    if (lower > task.upperscope()) { 
        task.update(m_config, task.upperscope(), upper, optimal_feature);
        return; 
    } // ^ similar reason to check in load_children. Don't know exact lb, 
      // just that it exceeds the upperscope()
    task.update(m_config, lower, upper, optimal_feature);
}

void Optimizer::link_to_parent(Bitmask const &parent, Bitmask const &features, Bitmask const &signs, float scope,
                               Bitmask const &self, translation_type const &order, adjacency_accessor &parents) {
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            int feature = (signs.get(j) ? 1 : -1) * (j + 1);
            m_graph.translations.insert(std::make_pair(std::make_pair(parent, feature), order));  // insert translation
            m_graph.children.insert(std::make_pair(std::make_pair(parent, feature),
                                                   self));  // insert forward look-up entry
            m_graph.edges.insert(parents,
                                 self);  // insert backward look-up entry
            std::pair<adjacency_iterator, bool> insertion = parents->second.insert(
                std::make_pair(parent, std::make_pair(Bitmask(m_dataset.m_number_features, false), scope)));
            insertion.first->second.first.set(j, true);
            insertion.first->second.second = std::min(insertion.first->second.second, scope);
        }
    }
}

void Optimizer::signal_exploiters(adjacency_accessor &parents, Task &self, unsigned int id) {
    if (self.uncertainty() != 0 && self.lowerbound() < self.lowerscope() - std::numeric_limits<float>::epsilon()) {
        return;
    }
    for (adjacency_iterator iterator = parents->second.begin(); iterator != parents->second.end(); ++iterator) {
        if (iterator->second.first.count() == 0) {
            continue;
        }
        if (self.lowerbound() < iterator->second.second - std::numeric_limits<float>::epsilon() &&
            self.uncertainty() > 0) {
            continue;
        }
        m_local_states[id].outbound_message.exploitation(self.capture_set(),                   // sender tile
                                                         iterator->first,                      // recipient tile
                                                         iterator->second.first,               // recipient features
                                                         self.support() - self.lowerbound());  // priority
        m_queue.push(m_local_states[id].outbound_message);
    }
}

bool Optimizer::update_root(float lower, float upper) {
    bool change = lower != this->global_lowerbound || upper != this->global_upperbound;
    this->global_lowerbound = lower;
    this->global_upperbound = upper;
    this->global_lowerbound = std::min(this->global_upperbound, this->global_lowerbound);
    this->global_boundary = global_upperbound - global_lowerbound;
    return change;
}
