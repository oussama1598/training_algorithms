#include "pla.h"

namespace {
    TEST_CASE("Testing the training of the pla") {
        DataSet dataset = DataSet::from_csv(
                "datasets/simple_non_noisy_data.csv",
                {"x1", "x2"},
                "y"
        );
        PLA pla{dataset};
        Metrics metrics = Metrics::load(dataset, pla);

        pla.train();

        REQUIRE(pla.predict({2.0, 1.0, 1.0}) == 1);
        REQUIRE(pla.predict({4.0, 5.0, 1.0}) == -1);

        metrics.save("training_output/pla_training_evolution.json");
    }
}
