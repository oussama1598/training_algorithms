#include "regression.h"

namespace {
    TEST_CASE("Testing the training of the regression") {
        std::vector<std::vector<double>> inputs;
        std::vector<double> labels;

        io::CSVReader<2> in("regression.csv");
        in.read_header(io::ignore_extra_column, "x", "y");

        double x;
        double y;

        while (in.read_row(x, y)) {
            inputs.push_back({x});
            labels.push_back(y);
        }


        Regression regression{inputs, labels};

        regression.train();

//        REQUIRE(perceptron.predict({2.0, 1.0, 1.0}) == 1);
//        REQUIRE(perceptron.predict({4.0, 5.0, 1.0}) == -1);

        json j;

        j["data"] = inputs;
        j["labels"] = labels;
        j["weights"] = regression.get_weight_history();
        j["losses"] = regression.get_losses_history();

        save_to_file("regression_training_evolution.json", j.dump(3));
    }
}
