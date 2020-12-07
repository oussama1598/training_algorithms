#include "logistic_regression.h"

namespace {
    TEST_CASE("Testing the training of the logistic linear_regression") {
        DataSet dataset = DataSet::from_csv(
                "datasets/simple_logistic_noisy_data.csv",
                {"x1", "x2"},
                "y"
        );

        LogisticRegression logistic_regression{
                dataset,
                1000,
                0.01
        };

        logistic_regression.train();

        json j;

        j["data"] = dataset.get_inputs();
        j["labels"] = dataset.get_labels();
        j["weights"] = logistic_regression.get_weight_history();
        j["losses"] = logistic_regression.get_losses_history();

        save_to_file("training_output/logistic_regression_training_evolution.json", j.dump(3));
    }
}
