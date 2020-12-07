#include "pla.h"

namespace {
    TEST_CASE("Testing the training of the pla") {
        DataSet dataset = DataSet::from_csv(
                "datasets/simple_non_noisy_data.csv",
                {"x1", "x2"},
                "y"
        );
        PLA pla{dataset};

        pla.train();

        REQUIRE(pla.predict({2.0, 1.0, 1.0}) == 1);
        REQUIRE(pla.predict({4.0, 5.0, 1.0}) == -1);

        json j;

        j["data"] = dataset.get_inputs();
        j["labels"] = dataset.get_labels();
        j["weights"] = pla.get_weight_history();
        j["losses"] = pla.get_losses_history();

        save_to_file("training_output/pla_training_evolution.json", j.dump(3));
    }
}
