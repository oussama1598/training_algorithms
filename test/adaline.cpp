#include "adaline.h"

namespace {
    TEST_CASE("Testing the training of the adaline") {
        DataSet dataset = DataSet::from_csv(
                "datasets/simple_noisy_data.csv",
                {"x1", "x2"},
                "y"
        );

        Adaline adaline{
                dataset,
                1000
        };
        Metrics metrics = Metrics::load(dataset, adaline);

        adaline.train();

        REQUIRE(adaline.predict({0.0, 2.0, 1.0}) == 1);
        REQUIRE(adaline.predict({5.0, 4.0, 1.0}) == -1);

        metrics.save("training_output/adaline_training_evolution.json");
    }
}
