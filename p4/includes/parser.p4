#ifndef KVSWITCH_PARSER_P4
#define KVSWITCH_PARSER_P4

const bit<16> TYPE_IPV4 = 0x800;
const bit<8>  TYPE_UDP  = 0x11;

parser KVSwitchParser(
    packet_in packet,
    out headers_t hdr,
    inout metadata_t meta,
    inout standard_metadata_t standard_metadata
) {
    state start {
        transition parse_ethernet;
    }

    state parse_ethernet {
        packet.extract(hdr.ethernet);
        transition select(hdr.ethernet.etherType) {
            TYPE_IPV4: parse_ipv4;
            default: accept;
        }
    }

    state parse_ipv4 {
        packet.extract(hdr.ipv4);
        transition select(hdr.ipv4.protocol) {
            TYPE_UDP: parse_udp;
            default: accept;
        }
    }

    state parse_udp {
        packet.extract(hdr.udp);
        transition select(hdr.udp.dstPort) {
            KVSWITCH_UDP_PORT: parse_kvswitch;
            default: accept;
        }
    }

    state parse_kvswitch {
        packet.extract(hdr.kvswitch);
        transition accept;
    }
}

#endif
