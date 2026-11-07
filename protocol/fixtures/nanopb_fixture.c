/* File: nanopb_fixture.c
 * Purpose: Host conformance only: independent identity encoding and shared-fixture roundtrips.
 * Contributors: Unassigned
 */
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
#include "pb_encode.h"
#include "pb_decode.h"
#include "orientation.pb.h"
#include "telemetry.pb.h"

int main(int argc, char **argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    uint8_t input[154], output[153];
    telemetry_orientation_v1_OrientationPacket packet = telemetry_orientation_v1_OrientationPacket_init_zero;
    telemetry_ImuStripData raw = telemetry_ImuStripData_init_zero;
    const pb_msgdesc_t *fields = telemetry_orientation_v1_OrientationPacket_fields;
    void *message = &packet;
    if (argc != 2) { return 2; }
    if (strcmp(argv[1], "emit") == 0) {
        packet.has_version = packet.has_sequence = packet.has_sample_time_ms = true;
        packet.version = 1;
        packet.orientations_count = 4;
        for (size_t i = 0; i < 4; ++i) {
            packet.orientations[i].has_sensor_id = packet.orientations[i].has_valid = true;
            packet.orientations[i].sensor_id = (uint32_t)i + 1;
            packet.orientations[i].valid = true;
            packet.orientations[i].has_rotation = true;
            packet.orientations[i].rotation.w = 1;
        }
    } else {
        if (strcmp(argv[1], "raw") == 0) { fields = telemetry_ImuStripData_fields; message = &raw; }
        size_t length = fread(input, 1, sizeof(input), stdin);
        if (length > sizeof(output)) { return 3; }
        pb_istream_t stream = pb_istream_from_buffer(input, length);
        if (!pb_decode(&stream, fields, message)) { fprintf(stderr, "%s\n", PB_GET_ERROR(&stream)); return 4; }
    }
    pb_ostream_t stream = pb_ostream_from_buffer(output, sizeof(output));
    if (!pb_encode(&stream, fields, message)) { fprintf(stderr, "%s\n", PB_GET_ERROR(&stream)); return 5; }
    return fwrite(output, 1, stream.bytes_written, stdout) == stream.bytes_written ? 0 : 6;
}
