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

        Metrics metrics = Metrics::load(dataset, regression);

        regression.train();

        metrics.save("training_output/linear_regression_training_evolution.json");
    }
}
