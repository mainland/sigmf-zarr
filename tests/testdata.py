"""Small, independently defined SigMF fixtures used by the test suite."""

import numpy as np
from sigmf import SigMFFile, __specification__

TEST_FLOAT32_DATA = np.arange(16, dtype=np.float32)
TEST_METADATA = {
    SigMFFile.ANNOTATION_KEY: [
        {"core:sample_count": 16, "core:sample_start": 0}
    ],
    SigMFFile.CAPTURE_KEY: [{"core:sample_start": 0}],
    SigMFFile.GLOBAL_KEY: {
        "core:datatype": "rf32_le",
        "core:sha512": (
            "f4984219b318894fa7144519185d1ae81ea721c6113243a52b51e444512a"
            "39d74cf41a4cec3c5d000bd7277cc71232c04d7a946717497e18619bdbe"
            "94bfeadd6"
        ),
        "core:num_channels": 1,
        "core:offset": 0,
        "core:version": __specification__,
    },
}
