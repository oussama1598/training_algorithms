#include "adaline.h"

namespace {
    TEST_CASE("Testing the training of the adaline") {
        std::vector<std::vector<double>> inputs{
                {0.0, 2.0},
                {1.0, 0.0},
                {1.0, 1.0},
                {1.0, 2.0},
                {1.0, 3.0},
                {1.0, 3.5},
                {2.0, 1.0},
                {2.0, 2.0},
                {2.0, 3.0},
                {2.0, 3.5},
                {3.0, 0.0},
                {3.0, 2.0},
                {2.7, 3.3},
                {4.0, 1.0},
                {1.0, 4.0},
                {2.0, 4.0},
                {2.0, 5.0},
                {2.5, 3.0},
                {2.5, 4.0},
                {3.0, 3.0},
                {3.0, 4.5},
                {3.0, 6.0},
                {4.0, 2.5},
                {4.0, 3.5},
                {4.0, 5.0},
                {5.0, 2.0},
                {5.0, 3.0},
                {5.0, 4.0},
        };

        std::vector<double> labels{
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0,
                -1.0
        };


        Adaline adaline{inputs, labels};

        adaline.train();

        REQUIRE(adaline.predict({0.0, 2.0, 1.0}) == 1);
        REQUIRE(adaline.predict({5.0, 4.0, 1.0}) == -1);

        json j;

        j["data"] = inputs;
        j["labels"] = labels;
        j["weights"] = adaline.get_weight_history();
        j["losses"] = adaline.get_losses_history();

        save_to_file("adaline_training_evolution.json", j.dump(3));
    }
}
