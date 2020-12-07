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
        Metrics metrics = Metrics::load(dataset, logistic_regression);

        logistic_regression.train();

        metrics.save("training_output/logistic_regression_training_evolution.json");
    }
}
