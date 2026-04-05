#ifndef KVSWITCH_HEADERS_P4
#define KVSWITCH_HEADERS_P4

const bit<16> KVSWITCH_UDP_PORT = 4789;
const bit<16> ECMP_BUCKET_COUNT = 32;

header ethernet_t {
    bit<48> dstAddr;
    bit<48> srcAddr;
    bit<16> etherType;
}

header ipv4_t {
    bit<4> version;
    bit<4> ihl;
    bit<8> diffserv;
    bit<16> totalLen;
    bit<16> identification;
    bit<3> flags;
    bit<13> fragOffset;
    bit<8> ttl;
    bit<8> protocol;
    bit<16> hdrChecksum;
    bit<32> srcAddr;
    bit<32> dstAddr;
}

header udp_t {
    bit<16> srcPort;
    bit<16> dstPort;
    bit<16> len;
    bit<16> checksum;
}

header kvswitch_shim_t {
    bit<4> version;
    bit<4> n_chunks;
    bit<8> flags;
    bit<16> req_id;
    bit<32> h0;
    bit<32> h1;
    bit<32> h2;
    bit<32> h3;
}

struct headers_t {
    ethernet_t ethernet;
    ipv4_t ipv4;
    udp_t udp;
    kvswitch_shim_t kvswitch;
}

struct metadata_t {
    bit<16> ecmp_bucket;
    bit<16> leaf_ecmp_bucket;
    bit<16> prefix_ecmp_group;
    bit<16> leaf_prefix_ecmp_group;
}

#endif
