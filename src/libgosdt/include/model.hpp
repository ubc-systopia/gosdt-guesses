#ifndef MODEL_H
#define MODEL_H

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>

#include "configuration.hpp"
#include "dataset.hpp"
#include "graph.hpp"
#include "nlohmann/json.hpp"

using json = nlohmann::json;

// Container for holding classification model extracted from the dependency
// graph
class Model {
   public:
    Model(void);
    // Constructor for terminal node in a model
    // @param set: shared pointer to a bitmask that identifies the captured set
    // of data points
    Model(std::shared_ptr<Bitmask> set, const Dataset &dataset, Bitmask &work_buffer);

    // Constructor for non-terminal node in a model
    // @param binary_feature_index: the index of the feature used for splitting
    // (after encoding)
    // @param negative: shared pointer to the model acting as the left subtree
    // @param positive: shared pointer to the model acting as the right subtree
    Model(unsigned int binary_feature_index, std::shared_ptr<Model> negative, std::shared_ptr<Model> positive,
          const Dataset &dataset);

    ~Model(void);

    // Hash generated from the leaf set of model
    size_t const hash(void) const;

    void identify(key_type const &indentifier);
    bool identified(void);

    void translate_self(translation_type const &translation);
    void translate_negatives(translation_type const &translation);
    void translate_positives(translation_type const &translation);

    // Equality operator implemented by comparing the set of addresses of the
    // bitmask of each leaf
    // @param other: other model to compare against
    // @returns true if the two models are provably equivalent
    // @note the equality comparison assumes that leaf bitmasks are not
    // duplicated
    //       this assumes that identical bitmasks are only copy by reference,
    //       not by value
    bool const operator==(Model const &other) const;

    // @param sample: bitmask of binary features (encoded) used to make the
    // prediction
    // @modifies prediction: string representation of the class that is
    // predicted
    void predict(Bitmask const &sample, std::string &prediction) const;

    // @returns: the training loss incurred by this model
    float loss(void) const;

    // @returns: the complexity penalty incurred by this model
    float complexity(void) const;

    // @modifies node: JSON object representation of this model
    void to_json(json &node, const Dataset &dataset) const;
    void _to_json(json &node, const Dataset &dataset) const;

    void translate_json(json &node, translation_type const &main, translation_type const &alternative,
                        unsigned int n_features) const;

    void summarize(json &node) const;
    void intersect(json &src, json &dest) const;

    // @param spacing: number of spaces to used in the indentation format
    // @modifies serialization: string representation of the JSON object
    // representation of this model
    void serialize(std::string &serialization, const Dataset &dataset, int const spacing = 0) const;

    key_type identifier;  // Identifier for association to graph vertex

    bool terminal = false;  // Flag specifying whether the node is terminal
   private:
    // Addresses of the bitmasks of the leaf set
    void _partitions(std::vector<Bitmask *> &addresses) const;
    void partitions(std::vector<Bitmask *> &addresses) const;

    // Non-terminal members
    unsigned int feature;                  // index of the decoded feature
    unsigned int binary_feature;           // index of the encoded feature
    unsigned int binary_target;            // index of the encoded target
    std::shared_ptr<Model> negative;       // left subtree
    std::shared_ptr<Model> positive;       // right subtree
    translation_type self_translator;      // self feature reordering
    translation_type negative_translator;  // left subtree feature reordering
    translation_type positive_translator;  // right subtree feature reordering

    // Terminal members
    std::string prediction;                // string representation of the predicted value
    float _loss;                           // loss incurred by this leaf
    float _complexity;                     // complexity penalty incurred by this leaf
    std::shared_ptr<Bitmask> capture_set;  // indicator specifying the points captured by this leaf
};

namespace std {
template <>
struct hash<Model> {
    std::size_t operator()(Model const &model) const { return model.hash(); }
};

template <>
struct hash<Model *> {
    std::size_t operator()(Model *const model) const { return model->hash(); }
};

template <>
struct equal_to<Model *> {
    bool operator()(Model *const left, Model *const right) const { return (*left) == (*right); }
};
}  // namespace std

namespace std {}

#endif