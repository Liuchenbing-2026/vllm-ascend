
#include "msa_dist_top_k_tiling.h"
#include "msa_dist_top_k.h"
#include "msa_dist_top_k_split.h"
#include <sstream>
#include <iostream>

namespace optiling {
namespace {

}

// MSA decode always uses the sequential split_s kernel (tiling key 10). The
// heuristic that hamming used to pick between key 1 and key 10 is dropped; this
// op only ships the key-10 path and the kernel only dispatches that path.
bool MsaDistTopKSplitSTiling::IsCapable() {
    SetPlatformInfoForTiling();
    return true;
}

uint64_t MsaDistTopKSplitSTiling::GetTilingKey() {
    return 10;
}

}
