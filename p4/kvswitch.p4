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

    action route_to_worker(bit<9> port, bit<48> dst_mac) {
        standard_metadata.egress_spec = port;
        hdr.ethernet.dstAddr = dst_mac;
    }

    action drop() {
        mark_to_drop(standard_metadata);
    }

    action forward(bit<9> port) {
        standard_metadata.egress_spec = port;
    }

    action set_prefix_ecmp_group(bit<16> group_id) {
        meta.prefix_ecmp_group = group_id;
    }

    action set_leaf_prefix_ecmp_group(bit<16> group_id) {
        meta.leaf_prefix_ecmp_group = group_id;
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

    // Independent leaf-tier hash using crc32 so leaf ECMP decisions are
    // decorrelated from spine-tier bucket assignments.
    action compute_leaf_ecmp_bucket() {
        hash(
            meta.leaf_ecmp_bucket,
            HashAlgorithm.crc32,
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

    // Spine prefix match: maps h0 to a per-prefix ECMP group ID.
    table spine_prefix_route {
        key = {
            hdr.kvswitch.h0: ternary;
        }
        actions = {
            set_prefix_ecmp_group;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    // Per-prefix ECMP: distributes prefix-matched traffic across leaves
    // weighted by cache locality and load.
    table spine_prefix_ecmp {
        key = {
            meta.prefix_ecmp_group: exact;
            meta.ecmp_bucket: exact;
        }
        actions = {
            route_to_pod;
            drop;
        }
        size = 4096;
        default_action = drop();
    }

    // Leaf prefix match: maps (h0, h1, h2) to a per-prefix leaf ECMP group.
    table leaf_prefix_route {
        key = {
            hdr.kvswitch.h0: ternary;
            hdr.kvswitch.h1: ternary;
            hdr.kvswitch.h2: ternary;
        }
        actions = {
            set_leaf_prefix_ecmp_group;
            NoAction;
        }
        size = 1024;
        default_action = NoAction();
    }

    // Per-prefix leaf ECMP: distributes prefix-matched traffic across
    // local workers weighted by cache locality and load.
    table leaf_prefix_ecmp {
        key = {
            meta.leaf_prefix_ecmp_group: exact;
            meta.leaf_ecmp_bucket: exact;
        }
        actions = {
            route_to_worker;
            drop;
        }
        size = 4096;
        default_action = drop();
    }

    // Miss-path ECMP: distributes non-prefix-matched traffic.
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
            meta.leaf_ecmp_bucket: exact;
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
            // Compute both ECMP buckets up front — spine and leaf use
            // independent hashes to decorrelate tier decisions.
            compute_ecmp_bucket();
            compute_leaf_ecmp_bucket();

            if (leaf_prefix_route.apply().hit) {
                // Per-prefix leaf ECMP across local cached workers.
                leaf_prefix_ecmp.apply();
            } else if (spine_prefix_route.apply().hit) {
                // Per-prefix spine ECMP across cached leaves.
                spine_prefix_ecmp.apply();
            } else {
                // Miss path: load-balanced ECMP fallback.
                if (!leaf_ecmp_select.apply().hit) {
                    spine_ecmp_select.apply();
                }
            }
        } else if (hdr.ipv4.isValid()) {
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
