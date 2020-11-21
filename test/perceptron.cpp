#include "perceptron.h"

namespace {
    TEST_CASE("Testing the training of the perceptron") {
        std::vector<std::vector<double>> inputs{
                {1.0, 1.0},
                {2.0, 1.0},
                {2.0, 2.0},
                {1.0, 3.0},
                {3.0, 3.0},
                {2.0, 4.0},
                {5.0, 4.0},
                {4.0, 5.0}
        };

        std::vector<double> labels{
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0
        };


        Perceptron perceptron{inputs, labels};

        perceptron.train();

        REQUIRE(perceptron.predict({2.0, 1.0, 1.0}) == 1);
        REQUIRE(perceptron.predict({4.0, 5.0, 1.0}) == -1);

        json j;

        j["data"] = inputs;
        j["labels"] = labels;
        j["weights"] = perceptron.get_weight_history();
        j["losses"] = perceptron.get_losses_history();

        save_to_file("perceptron_training_evolution.json", j.dump(3));
    }
}
