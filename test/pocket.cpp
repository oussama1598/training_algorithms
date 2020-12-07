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

        pocket.train();

        REQUIRE(pocket.predict({0.0, 2.0, 1.0}) == 1);
        REQUIRE(pocket.predict({5.0, 4.0, 1.0}) == -1);

        json j;

        j["data"] = dataset.get_inputs();
        j["labels"] = dataset.get_labels();
        j["weights"] = pocket.get_weight_history();
        j["losses"] = pocket.get_losses_history();

        save_to_file("training_output/pocket_training_evolution.json", j.dump(3));
    }
}
