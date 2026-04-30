from judicator.biases.authority import AuthorityBiasTest
from judicator.biases.concreteness import ConcretenessBiasTest
from judicator.biases.position import PositionBiasTest
from judicator.biases.scale_anchoring import ScaleAnchoringTest
from judicator.biases.self_consistency import SelfConsistencyTest
from judicator.biases.verbosity import VerbosityBiasTest
from judicator.biases.yes_bias import YesBiasTest

ALL_TESTS: list = [
    PositionBiasTest(),
    VerbosityBiasTest(),
    SelfConsistencyTest(),
    ScaleAnchoringTest(),
    AuthorityBiasTest(),
    ConcretenessBiasTest(),
    YesBiasTest(),
]
