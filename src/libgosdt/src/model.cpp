#include "model.hpp"

Model::Model(void) {}

Model::Model(std::shared_ptr<Bitmask> capture_set, const Dataset &dataset, Bitmask &work_buffer) {
    Dataset::SummaryStatistics stats = dataset.summary_statistics(*capture_set, work_buffer);
    this->binary_target = stats.optimal;
    this->_loss = stats.max_loss;
    this->_complexity = dataset.m_config.regularization;
    this->capture_set = capture_set;
    this->terminal = true;
}

Model::Model(unsigned int binary_feature_index, std::shared_ptr<Model> negative, std::shared_ptr<Model> positive,
             const Dataset &dataset) {
    std::string feature_name, feature_type, relation, reference;
    size_t feature_index = dataset.original_feature(binary_feature_index);

    this->binary_feature = binary_feature_index;
    this->feature = feature_index;
    this->negative = negative;
    this->positive = positive;
    this->terminal = false;
}

Model::~Model(void) {}

void Model::identify(Bitmask const &identifier) { this->identifier = identifier; }

bool Model::identified(void) { return this->identifier.size() > 0; }

void Model::translate_self(translation_type const &translation) { this->self_translator = translation; }

void Model::translate_negatives(translation_type const &translation) { this->negative_translator = translation; }

void Model::translate_positives(translation_type const &translation) { this->positive_translator = translation; }

void Model::_partitions(std::vector<Bitmask *> &addresses) const {
    if (this->terminal) {
        addresses.push_back(this->capture_set.get());
    } else {
        this->negative->_partitions(addresses);
        this->positive->_partitions(addresses);
    }

    return;
};

void Model::partitions(std::vector<Bitmask *> &sorted_addresses) const {
    std::vector<Bitmask *> addresses;
    _partitions(addresses);
    // std::cout << "_partition size: " << addresses.size() << std::endl;
    std::map<unsigned int, Bitmask *> sorted;
    // for (auto it = addresses.begin(); it != addresses.end(); ++it) {
    //     Bitmask * address = * it;
    //     std::cout << "Address: " << address << std::endl;

    //     unsigned int size = address -> size();
    //     for (unsigned int rank = 0; rank < size; ++rank) {
    //         if (address -> get(rank) == 1) {
    //             sorted[rank] = address;
    //             break;
    //         }
    //     }
    // }
    // for (auto it = sorted.begin(); it != sorted.end(); ++it) {
    //     sorted_addresses.push_back(it -> second);
    // }
    for (auto it = addresses.begin(); it != addresses.end(); ++it) {
        sorted_addresses.push_back(*it);
    }
    // std::cout << "partition size: " << sorted_addresses.size() << std::endl;
    return;
};

size_t const Model::hash(void) const {
    // std::cout << "hash functiion entry" << std::endl;

    std::vector<Bitmask *> addresses;
    partitions(addresses);
    size_t seed = addresses.size();
    // std::cout << "final partition size: " << addresses.size() << std::endl;
    for (auto it = addresses.begin(); it != addresses.end(); ++it) {
        // std::cout << "partition: " << (**it).to_string() << std::endl;
        seed ^= ((**it).hash()) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    // std::cout << "hash: " << seed << std::endl;
    return seed;
}

bool const Model::operator==(Model const &other) const {
    if (hash() != other.hash()) {
        return false;
    } else {
        std::vector<Bitmask *> masks;
        std::vector<Bitmask *> other_masks;
        partitions(masks);
        other.partitions(other_masks);
        if (masks.size() != other_masks.size()) {
            return false;
        }
        auto iterator = masks.begin();
        auto other_iterator = other_masks.begin();
        while (iterator != masks.end() && other_iterator != other_masks.end()) {
            if ((**iterator) != (**other_iterator)) {
                return false;
            }
            ++iterator;
            ++other_iterator;
        }
        return true;
    }
}

float Model::loss(void) const {
    // Currently a stub, need to implement
    if (this->terminal) {
        return this->_loss;
    } else {
        return this->negative->loss() + this->positive->loss();
    }
}

float Model::complexity(void) const {
    // Currently a stub, need to implement
    if (this->terminal) {
        return this->_complexity;
    } else {
        return this->negative->complexity() + this->positive->complexity();
    }
}

void Model::predict(Bitmask const &sample, std::string &prediction) const {
    // Currently a stub, need to implement
    return;
}

void Model::serialize(std::string &serialization, const Dataset &dataset, int const spacing) const {
    json node = json::object();
    to_json(node, dataset);
    serialization = spacing == 0 ? node.dump() : node.dump(spacing);
}

void Model::intersect(json &src, json &dest) const {
    if (!src[0].is_null() && !dest[0].is_null()) {
        dest[0] = std::max(src[0], dest[0]);
    } else if (!src[0].is_null() && dest[0].is_null()) {
        dest[0] = src[0];
    }
    if (!src[1].is_null() && !dest[1].is_null()) {
        dest[1] = std::min(src[1], dest[1]);
    } else if (!src[1].is_null() && dest[1].is_null()) {
        dest[1] = src[1];
    }
}

void Model::summarize(json &node) const {
    if (node.contains("feature")) {
        summarize(node["true"]);
        summarize(node["false"]);

        // Check feature domain type
        bool integral = node["type"] == "integral";
        bool rational = node["type"] == "rational";
        bool categorical = node["type"] == "categorical";

        node["children"] = {json::object(), json::object()};
        node["children"][0]["then"] = node["true"];
        node["children"][1]["then"] = node["false"];
        if (integral) {
            node["children"][0]["in"] = {node["reference"], nullptr};
            node["children"][1]["in"] = {nullptr, node["reference"]};
        } else if (rational) {
            node["children"][0]["in"] = {node["reference"], nullptr};
            node["children"][1]["in"] = {nullptr, node["reference"]};
        } else if (categorical) {
            node["children"][0]["in"] = node["reference"];
            node["children"][1]["in"] = "default";
        }
        node.erase("reference");
        node.erase("relation");
        node.erase("true");
        node.erase("false");

        json new_children = json::array();
        for (json::iterator it = node["children"].begin(); it != node["children"].end(); ++it) {
            json &condition = (*it)["in"];
            json &child = (*it)["then"];
            if (child.contains("feature") && child["feature"] == node["feature"]) {
                // Child has grand children and child feature matches parent
                // feature
                for (json::iterator sub_it = child["children"].begin(); sub_it != child["children"].end(); ++sub_it) {
                    json &subcondition = (*sub_it)["in"];
                    json &grandchild = (*sub_it)["then"];
                    if (integral || rational) {
                        // Promote grandchild into child
                        json promoted_condition = {subcondition[0], subcondition[1]};
                        intersect(condition, promoted_condition);
                        json promoted_child = {{"in", promoted_condition}, {"then", grandchild}};
                        new_children.push_back(promoted_child);
                    } else if (categorical) {
                        json promoted_child = {{"in", subcondition}, {"then", grandchild}};
                        new_children.push_back(promoted_child);
                    }
                }
            } else {  // re-insert
                json unpromoted_child = {{"in", condition}, {"then", child}};
                new_children.push_back(unpromoted_child);
            }
        }
        node["children"] = new_children;  // Overwrite previous list fo children
    } else {
        // Is a leaf node
        // No transformation
    }
}

void Model::to_json(json &node, const Dataset &dataset) const {
    _to_json(node, dataset);
    // Convert to N-ary
    if (dataset.m_config.non_binary) {
        summarize(node);
    }
}

void Model::_to_json(json &node, const Dataset &dataset) const {
    if (this->terminal) {
        node["prediction"] = this->binary_target;
        node["loss"] = this->_loss;  // This value is correct regardless of translation
        node["complexity"] = dataset.m_config.regularization;
    } else {
        node["feature"] = this->binary_feature;
        node["orig_feature"] = this->feature;
        node["false"] = json::object();
        node["true"] = json::object();
        this->negative->_to_json(node["false"], dataset);
        this->positive->_to_json(node["true"], dataset);

        if (this->negative_translator.size() > 0) {
            translate_json(node["false"], this->negative->self_translator, this->negative_translator,
                           dataset.m_number_features);
        }
        if (this->positive_translator.size() > 0) {
            translate_json(node["true"], this->positive->self_translator, this->positive_translator,
                           dataset.m_number_features);
        }
    }
}

void Model::translate_json(json &node, translation_type const &main, translation_type const &alternative,
                           unsigned int n_features) const {
    if (node.contains("prediction")) {
        // index translation to undo any reordering from tile normalization
        int cannonical_index = (int)(node["prediction"]) + n_features;
        int normal_index = std::distance(main.begin(), std::find(main.begin(), main.end(), cannonical_index));
        int alternative_index = (int)(alternative.at(normal_index)) - n_features;

        node["prediction"] = alternative_index;
    } else if (node.contains("feature")) {
        // index translation to undo any reordering from tile normalization
        bool flip = false;
        int cannonical_index = node["feature"];
        int normal_index;
        if (std::find(main.begin(), main.end(), cannonical_index) != main.end()) {
            normal_index = std::distance(main.begin(), std::find(main.begin(), main.end(), cannonical_index));
        } else if (std::find(main.begin(), main.end(), -cannonical_index) != main.end()) {
            normal_index = std::distance(main.begin(), std::find(main.begin(), main.end(), -cannonical_index));
            flip = !flip;
        }
        int alternative_index = alternative.at(normal_index);
        if (alternative_index < 0) {
            flip = !flip;
        }

        node["feature"] = std::abs(alternative_index);
        translate_json(node["false"], main, alternative, n_features);
        translate_json(node["true"], main, alternative, n_features);
        if (flip) {
            node["swap"] = node["true"];
            node["true"] = node["false"];
            node["false"] = node["swap"];
            node.erase("swap");
        }
    }
}
