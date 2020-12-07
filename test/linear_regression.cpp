#include "linear_regression.h"

namespace {
    TEST_CASE("Testing the training of the linear_regression") {
        DataSet dataset = DataSet::from_csv(
                "datasets/multi_dimensional_regression_data.csv",
                {"X1", "X2", "X3", "X4"},
                "X5"
        );

        LinearRegression regression{
                dataset,
                1000,
                0.0000001
        };

        regression.train();

        json j;

        j["data"] = dataset.get_inputs();
        j["labels"] = dataset.get_labels();
        j["weights"] = regression.get_weight_history();
        j["losses"] = regression.get_losses_history();

        save_to_file("training_output/linear_regression_training_evolution.json", j.dump(3));
    }
}
