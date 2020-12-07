#include "pocket.h"

namespace {
    TEST_CASE("Testing the training of the pocket") {
        DataSet dataset = DataSet::from_csv(
                "datasets/simple_noisy_data.csv",
                {"x1", "x2"},
                "y"
        );

        Pocket pocket{
                dataset,
                1000
        };

        Metrics metrics = Metrics::load(dataset, pocket);

        pocket.train();

        REQUIRE(pocket.predict({0.0, 2.0, 1.0}) == 1);
        REQUIRE(pocket.predict({5.0, 4.0, 1.0}) == -1);


        metrics.save("training_output/pocket_training_evolution.json");
    }
}
