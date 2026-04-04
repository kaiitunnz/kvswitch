#include <core.p4>
#include <v1model.p4>

#include "includes/headers.p4"
#include "includes/parser.p4"

control KVSwitchVerifyChecksum(
    inout headers_t hdr,
    inout metadata_t meta
) {
    apply { }
}

control KVSwitchIngress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_metadata
) {
    action route_to_pod(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action route_to_worker(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action forward(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action compute_ecmp_bucket() {
        hash(
            meta.ecmp_bucket,
            HashAlgorithm.crc16,
            (bit<16>)0,
            {
                hdr.ipv4.srcAddr,
                hdr.ipv4.dstAddr,
                hdr.udp.srcPort,
                hdr.udp.dstPort,
                hdr.kvswitch.req_id
            },
            ECMP_BUCKET_COUNT
        );
    }

    // Basic IPv4 forwarding for non-KVSwitch traffic (ARP, ping, etc.).
    table ipv4_lpm {
        key = {
            hdr.ipv4.dstAddr: lpm;
        }
        actions = {
            forward;
            drop;
            NoAction;
        }
        size = 256;
        default_action = NoAction();
    }

    table spine_prefix_route {
        key = {
            hdr.kvswitch.h0: ternary;
        }
        actions = {
            route_to_pod;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table leaf_prefix_route {
        key = {
            hdr.kvswitch.h0: ternary;
            hdr.kvswitch.h1: ternary;
            hdr.kvswitch.h2: ternary;
        }
        actions = {
            route_to_worker;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    table spine_ecmp_select {
        key = {
            meta.ecmp_bucket: exact;
        }
        actions = {
            route_to_pod;
            drop;
        }
        size = ECMP_BUCKET_COUNT;
        default_action = drop();
    }

    table leaf_ecmp_select {
        key = {
            meta.ecmp_bucket: exact;
        }
        actions = {
            route_to_worker;
            drop;
        }
        size = ECMP_BUCKET_COUNT;
        default_action = drop();
    }

    apply {
        if (hdr.kvswitch.isValid()) {
            // KVSwitch shim header present — use prefix-aware routing.
            if (!leaf_prefix_route.apply().hit) {
                if (!spine_prefix_route.apply().hit) {
                    compute_ecmp_bucket();
                    if (!leaf_ecmp_select.apply().hit) {
                        spine_ecmp_select.apply();
                    }
                }
            }
        } else if (hdr.ipv4.isValid()) {
            // Regular IPv4 traffic — basic LPM forwarding.
            ipv4_lpm.apply();
        }
    }
}

control KVSwitchEgress(
    inout headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_metadata
) {
    apply { }
}

control KVSwitchComputeChecksum(
    inout headers_t hdr,
    inout metadata_t meta
) {
    apply { }
}

control KVSwitchDeparser(packet_out packet, in headers_t hdr) {
    apply {
        packet.emit(hdr.ethernet);
        packet.emit(hdr.ipv4);
        packet.emit(hdr.udp);
        packet.emit(hdr.kvswitch);
    }
}

V1Switch(
    KVSwitchParser(),
    KVSwitchVerifyChecksum(),
    KVSwitchIngress(),
    KVSwitchEgress(),
    KVSwitchComputeChecksum(),
    KVSwitchDeparser()
) main;
