#pragma once

// Build/toolchain sanity: reports the visible CUDA devices.
// Infra only (not part of the from-scratch ML core). Returns device count.
int report_cuda_devices();
