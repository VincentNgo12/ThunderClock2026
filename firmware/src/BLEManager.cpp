/**
 * File: BLEManager.cpp
 * Purpose: Unimplemented BLE setup and packet publication.
 * Contributors:
 * - Unassigned
 */
#include "BLEManager.h"

fw::Result bleBegin() {
  // TODO(FW-BLE): configure the agreed service, characteristics and connection diagnostics.
  return fw::Result::NotImplemented;
}

fw::Result blePublish(const uint8_t*, size_t, const uint8_t*, size_t) {
  // TODO(FW-BLE): verify subscription and negotiated capacity before sending complete packets.
  return fw::Result::NotImplemented;
}
