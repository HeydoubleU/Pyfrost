#include "python.h"
#include "toPython.h"
#include <unordered_map>

using ulong = unsigned long long;
using uint = unsigned int;
using ushort = unsigned short;
using uchar = unsigned char;


std::unordered_map<unsigned long long, int> TYPE_MAP = {
    { Amino::getTypeId<Amino::Ptr<Bifrost::Object>>().computeHash(), 1 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Amino::Ptr<Bifrost::Object>>>>().computeHash(), 2 },
    { Amino::getTypeId<Amino::String>().computeHash(), 3 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Amino::String>>>().computeHash(), 4 },
    { Amino::getTypeId<float>().computeHash(), 5 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<float>>>().computeHash(), 6 },
    { Amino::getTypeId<double>().computeHash(), 7 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<double>>>().computeHash(), 8 },
    { Amino::getTypeId<long long>().computeHash(), 9 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<long long>>>().computeHash(), 10 },
    { Amino::getTypeId<ulong>().computeHash(), 11 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<ulong>>>().computeHash(), 12 },
    { Amino::getTypeId<int>().computeHash(), 13 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<int>>>().computeHash(), 14 },
    { Amino::getTypeId<uint>().computeHash(), 15 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<uint>>>().computeHash(), 16 },
    { Amino::getTypeId<short>().computeHash(), 17 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<short>>>().computeHash(), 18 },
    { Amino::getTypeId<ushort>().computeHash(), 19 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<ushort>>>().computeHash(), 20 },
    { Amino::getTypeId<signed char>().computeHash(), 21 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<signed char>>>().computeHash(), 22 },
    { Amino::getTypeId<uchar>().computeHash(), 23 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<uchar>>>().computeHash(), 24 },
    { Amino::getTypeId<bool>().computeHash(), 25 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<bool>>>().computeHash(), 26 },
    { Amino::getTypeId<Bifrost::Math::float2>().computeHash(), 27 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float2>>>().computeHash(), 28 },
    { Amino::getTypeId<Bifrost::Math::float3>().computeHash(), 29 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float3>>>().computeHash(), 30 },
    { Amino::getTypeId<Bifrost::Math::float4>().computeHash(), 31 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float4>>>().computeHash(), 32 },
    { Amino::getTypeId<Bifrost::Math::float2x2>().computeHash(), 33 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float2x2>>>().computeHash(), 34 },
    { Amino::getTypeId<Bifrost::Math::float2x3>().computeHash(), 35 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float2x3>>>().computeHash(), 36 },
    { Amino::getTypeId<Bifrost::Math::float2x4>().computeHash(), 37 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float2x4>>>().computeHash(), 38 },
    { Amino::getTypeId<Bifrost::Math::float3x2>().computeHash(), 39 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float3x2>>>().computeHash(), 40 },
    { Amino::getTypeId<Bifrost::Math::float3x3>().computeHash(), 41 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float3x3>>>().computeHash(), 42 },
    { Amino::getTypeId<Bifrost::Math::float3x4>().computeHash(), 43 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float3x4>>>().computeHash(), 44 },
    { Amino::getTypeId<Bifrost::Math::float4x2>().computeHash(), 45 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float4x2>>>().computeHash(), 46 },
    { Amino::getTypeId<Bifrost::Math::float4x3>().computeHash(), 47 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float4x3>>>().computeHash(), 48 },
    { Amino::getTypeId<Bifrost::Math::float4x4>().computeHash(), 49 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::float4x4>>>().computeHash(), 50 },
    { Amino::getTypeId<Bifrost::Math::double2>().computeHash(), 51 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double2>>>().computeHash(), 52 },
    { Amino::getTypeId<Bifrost::Math::double3>().computeHash(), 53 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double3>>>().computeHash(), 54 },
    { Amino::getTypeId<Bifrost::Math::double4>().computeHash(), 55 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double4>>>().computeHash(), 56 },
    { Amino::getTypeId<Bifrost::Math::double2x2>().computeHash(), 57 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double2x2>>>().computeHash(), 58 },
    { Amino::getTypeId<Bifrost::Math::double2x3>().computeHash(), 59 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double2x3>>>().computeHash(), 60 },
    { Amino::getTypeId<Bifrost::Math::double2x4>().computeHash(), 61 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double2x4>>>().computeHash(), 62 },
    { Amino::getTypeId<Bifrost::Math::double3x2>().computeHash(), 63 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double3x2>>>().computeHash(), 64 },
    { Amino::getTypeId<Bifrost::Math::double3x3>().computeHash(), 65 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double3x3>>>().computeHash(), 66 },
    { Amino::getTypeId<Bifrost::Math::double3x4>().computeHash(), 67 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double3x4>>>().computeHash(), 68 },
    { Amino::getTypeId<Bifrost::Math::double4x2>().computeHash(), 69 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double4x2>>>().computeHash(), 70 },
    { Amino::getTypeId<Bifrost::Math::double4x3>().computeHash(), 71 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double4x3>>>().computeHash(), 72 },
    { Amino::getTypeId<Bifrost::Math::double4x4>().computeHash(), 73 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::double4x4>>>().computeHash(), 74 },
    { Amino::getTypeId<Bifrost::Math::long2>().computeHash(), 75 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long2>>>().computeHash(), 76 },
    { Amino::getTypeId<Bifrost::Math::long3>().computeHash(), 77 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long3>>>().computeHash(), 78 },
    { Amino::getTypeId<Bifrost::Math::long4>().computeHash(), 79 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long4>>>().computeHash(), 80 },
    { Amino::getTypeId<Bifrost::Math::long2x2>().computeHash(), 81 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long2x2>>>().computeHash(), 82 },
    { Amino::getTypeId<Bifrost::Math::long2x3>().computeHash(), 83 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long2x3>>>().computeHash(), 84 },
    { Amino::getTypeId<Bifrost::Math::long2x4>().computeHash(), 85 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long2x4>>>().computeHash(), 86 },
    { Amino::getTypeId<Bifrost::Math::long3x2>().computeHash(), 87 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long3x2>>>().computeHash(), 88 },
    { Amino::getTypeId<Bifrost::Math::long3x3>().computeHash(), 89 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long3x3>>>().computeHash(), 90 },
    { Amino::getTypeId<Bifrost::Math::long3x4>().computeHash(), 91 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long3x4>>>().computeHash(), 92 },
    { Amino::getTypeId<Bifrost::Math::long4x2>().computeHash(), 93 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long4x2>>>().computeHash(), 94 },
    { Amino::getTypeId<Bifrost::Math::long4x3>().computeHash(), 95 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long4x3>>>().computeHash(), 96 },
    { Amino::getTypeId<Bifrost::Math::long4x4>().computeHash(), 97 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::long4x4>>>().computeHash(), 98 },
    { Amino::getTypeId<Bifrost::Math::ulong2>().computeHash(), 99 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2>>>().computeHash(), 100 },
    { Amino::getTypeId<Bifrost::Math::ulong3>().computeHash(), 101 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3>>>().computeHash(), 102 },
    { Amino::getTypeId<Bifrost::Math::ulong4>().computeHash(), 103 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4>>>().computeHash(), 104 },
    { Amino::getTypeId<Bifrost::Math::ulong2x2>().computeHash(), 105 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2x2>>>().computeHash(), 106 },
    { Amino::getTypeId<Bifrost::Math::ulong2x3>().computeHash(), 107 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2x3>>>().computeHash(), 108 },
    { Amino::getTypeId<Bifrost::Math::ulong2x4>().computeHash(), 109 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2x4>>>().computeHash(), 110 },
    { Amino::getTypeId<Bifrost::Math::ulong3x2>().computeHash(), 111 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3x2>>>().computeHash(), 112 },
    { Amino::getTypeId<Bifrost::Math::ulong3x3>().computeHash(), 113 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3x3>>>().computeHash(), 114 },
    { Amino::getTypeId<Bifrost::Math::ulong3x4>().computeHash(), 115 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3x4>>>().computeHash(), 116 },
    { Amino::getTypeId<Bifrost::Math::ulong4x2>().computeHash(), 117 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4x2>>>().computeHash(), 118 },
    { Amino::getTypeId<Bifrost::Math::ulong4x3>().computeHash(), 119 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4x3>>>().computeHash(), 120 },
    { Amino::getTypeId<Bifrost::Math::ulong4x4>().computeHash(), 121 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4x4>>>().computeHash(), 122 },
    { Amino::getTypeId<Bifrost::Math::int2>().computeHash(), 123 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int2>>>().computeHash(), 124 },
    { Amino::getTypeId<Bifrost::Math::int3>().computeHash(), 125 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int3>>>().computeHash(), 126 },
    { Amino::getTypeId<Bifrost::Math::int4>().computeHash(), 127 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int4>>>().computeHash(), 128 },
    { Amino::getTypeId<Bifrost::Math::int2x2>().computeHash(), 129 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int2x2>>>().computeHash(), 130 },
    { Amino::getTypeId<Bifrost::Math::int2x3>().computeHash(), 131 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int2x3>>>().computeHash(), 132 },
    { Amino::getTypeId<Bifrost::Math::int2x4>().computeHash(), 133 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int2x4>>>().computeHash(), 134 },
    { Amino::getTypeId<Bifrost::Math::int3x2>().computeHash(), 135 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int3x2>>>().computeHash(), 136 },
    { Amino::getTypeId<Bifrost::Math::int3x3>().computeHash(), 137 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int3x3>>>().computeHash(), 138 },
    { Amino::getTypeId<Bifrost::Math::int3x4>().computeHash(), 139 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int3x4>>>().computeHash(), 140 },
    { Amino::getTypeId<Bifrost::Math::int4x2>().computeHash(), 141 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int4x2>>>().computeHash(), 142 },
    { Amino::getTypeId<Bifrost::Math::int4x3>().computeHash(), 143 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int4x3>>>().computeHash(), 144 },
    { Amino::getTypeId<Bifrost::Math::int4x4>().computeHash(), 145 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::int4x4>>>().computeHash(), 146 },
    { Amino::getTypeId<Bifrost::Math::uint2>().computeHash(), 147 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint2>>>().computeHash(), 148 },
    { Amino::getTypeId<Bifrost::Math::uint3>().computeHash(), 149 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint3>>>().computeHash(), 150 },
    { Amino::getTypeId<Bifrost::Math::uint4>().computeHash(), 151 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint4>>>().computeHash(), 152 },
    { Amino::getTypeId<Bifrost::Math::uint2x2>().computeHash(), 153 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint2x2>>>().computeHash(), 154 },
    { Amino::getTypeId<Bifrost::Math::uint2x3>().computeHash(), 155 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint2x3>>>().computeHash(), 156 },
    { Amino::getTypeId<Bifrost::Math::uint2x4>().computeHash(), 157 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint2x4>>>().computeHash(), 158 },
    { Amino::getTypeId<Bifrost::Math::uint3x2>().computeHash(), 159 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint3x2>>>().computeHash(), 160 },
    { Amino::getTypeId<Bifrost::Math::uint3x3>().computeHash(), 161 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint3x3>>>().computeHash(), 162 },
    { Amino::getTypeId<Bifrost::Math::uint3x4>().computeHash(), 163 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint3x4>>>().computeHash(), 164 },
    { Amino::getTypeId<Bifrost::Math::uint4x2>().computeHash(), 165 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint4x2>>>().computeHash(), 166 },
    { Amino::getTypeId<Bifrost::Math::uint4x3>().computeHash(), 167 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint4x3>>>().computeHash(), 168 },
    { Amino::getTypeId<Bifrost::Math::uint4x4>().computeHash(), 169 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uint4x4>>>().computeHash(), 170 },
    { Amino::getTypeId<Bifrost::Math::short2>().computeHash(), 171 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short2>>>().computeHash(), 172 },
    { Amino::getTypeId<Bifrost::Math::short3>().computeHash(), 173 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short3>>>().computeHash(), 174 },
    { Amino::getTypeId<Bifrost::Math::short4>().computeHash(), 175 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short4>>>().computeHash(), 176 },
    { Amino::getTypeId<Bifrost::Math::short2x2>().computeHash(), 177 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short2x2>>>().computeHash(), 178 },
    { Amino::getTypeId<Bifrost::Math::short2x3>().computeHash(), 179 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short2x3>>>().computeHash(), 180 },
    { Amino::getTypeId<Bifrost::Math::short2x4>().computeHash(), 181 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short2x4>>>().computeHash(), 182 },
    { Amino::getTypeId<Bifrost::Math::short3x2>().computeHash(), 183 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short3x2>>>().computeHash(), 184 },
    { Amino::getTypeId<Bifrost::Math::short3x3>().computeHash(), 185 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short3x3>>>().computeHash(), 186 },
    { Amino::getTypeId<Bifrost::Math::short3x4>().computeHash(), 187 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short3x4>>>().computeHash(), 188 },
    { Amino::getTypeId<Bifrost::Math::short4x2>().computeHash(), 189 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short4x2>>>().computeHash(), 190 },
    { Amino::getTypeId<Bifrost::Math::short4x3>().computeHash(), 191 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short4x3>>>().computeHash(), 192 },
    { Amino::getTypeId<Bifrost::Math::short4x4>().computeHash(), 193 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::short4x4>>>().computeHash(), 194 },
    { Amino::getTypeId<Bifrost::Math::ushort2>().computeHash(), 195 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2>>>().computeHash(), 196 },
    { Amino::getTypeId<Bifrost::Math::ushort3>().computeHash(), 197 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3>>>().computeHash(), 198 },
    { Amino::getTypeId<Bifrost::Math::ushort4>().computeHash(), 199 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4>>>().computeHash(), 200 },
    { Amino::getTypeId<Bifrost::Math::ushort2x2>().computeHash(), 201 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2x2>>>().computeHash(), 202 },
    { Amino::getTypeId<Bifrost::Math::ushort2x3>().computeHash(), 203 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2x3>>>().computeHash(), 204 },
    { Amino::getTypeId<Bifrost::Math::ushort2x4>().computeHash(), 205 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2x4>>>().computeHash(), 206 },
    { Amino::getTypeId<Bifrost::Math::ushort3x2>().computeHash(), 207 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3x2>>>().computeHash(), 208 },
    { Amino::getTypeId<Bifrost::Math::ushort3x3>().computeHash(), 209 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3x3>>>().computeHash(), 210 },
    { Amino::getTypeId<Bifrost::Math::ushort3x4>().computeHash(), 211 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3x4>>>().computeHash(), 212 },
    { Amino::getTypeId<Bifrost::Math::ushort4x2>().computeHash(), 213 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4x2>>>().computeHash(), 214 },
    { Amino::getTypeId<Bifrost::Math::ushort4x3>().computeHash(), 215 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4x3>>>().computeHash(), 216 },
    { Amino::getTypeId<Bifrost::Math::ushort4x4>().computeHash(), 217 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4x4>>>().computeHash(), 218 },
    { Amino::getTypeId<Bifrost::Math::char2>().computeHash(), 219 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char2>>>().computeHash(), 220 },
    { Amino::getTypeId<Bifrost::Math::char3>().computeHash(), 221 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char3>>>().computeHash(), 222 },
    { Amino::getTypeId<Bifrost::Math::char4>().computeHash(), 223 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char4>>>().computeHash(), 224 },
    { Amino::getTypeId<Bifrost::Math::char2x2>().computeHash(), 225 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char2x2>>>().computeHash(), 226 },
    { Amino::getTypeId<Bifrost::Math::char2x3>().computeHash(), 227 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char2x3>>>().computeHash(), 228 },
    { Amino::getTypeId<Bifrost::Math::char2x4>().computeHash(), 229 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char2x4>>>().computeHash(), 230 },
    { Amino::getTypeId<Bifrost::Math::char3x2>().computeHash(), 231 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char3x2>>>().computeHash(), 232 },
    { Amino::getTypeId<Bifrost::Math::char3x3>().computeHash(), 233 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char3x3>>>().computeHash(), 234 },
    { Amino::getTypeId<Bifrost::Math::char3x4>().computeHash(), 235 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char3x4>>>().computeHash(), 236 },
    { Amino::getTypeId<Bifrost::Math::char4x2>().computeHash(), 237 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char4x2>>>().computeHash(), 238 },
    { Amino::getTypeId<Bifrost::Math::char4x3>().computeHash(), 239 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char4x3>>>().computeHash(), 240 },
    { Amino::getTypeId<Bifrost::Math::char4x4>().computeHash(), 241 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::char4x4>>>().computeHash(), 242 },
    { Amino::getTypeId<Bifrost::Math::uchar2>().computeHash(), 243 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2>>>().computeHash(), 244 },
    { Amino::getTypeId<Bifrost::Math::uchar3>().computeHash(), 245 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3>>>().computeHash(), 246 },
    { Amino::getTypeId<Bifrost::Math::uchar4>().computeHash(), 247 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4>>>().computeHash(), 248 },
    { Amino::getTypeId<Bifrost::Math::uchar2x2>().computeHash(), 249 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2x2>>>().computeHash(), 250 },
    { Amino::getTypeId<Bifrost::Math::uchar2x3>().computeHash(), 251 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2x3>>>().computeHash(), 252 },
    { Amino::getTypeId<Bifrost::Math::uchar2x4>().computeHash(), 253 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2x4>>>().computeHash(), 254 },
    { Amino::getTypeId<Bifrost::Math::uchar3x2>().computeHash(), 255 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3x2>>>().computeHash(), 256 },
    { Amino::getTypeId<Bifrost::Math::uchar3x3>().computeHash(), 257 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3x3>>>().computeHash(), 258 },
    { Amino::getTypeId<Bifrost::Math::uchar3x4>().computeHash(), 259 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3x4>>>().computeHash(), 260 },
    { Amino::getTypeId<Bifrost::Math::uchar4x2>().computeHash(), 261 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4x2>>>().computeHash(), 262 },
    { Amino::getTypeId<Bifrost::Math::uchar4x3>().computeHash(), 263 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4x3>>>().computeHash(), 264 },
    { Amino::getTypeId<Bifrost::Math::uchar4x4>().computeHash(), 265 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4x4>>>().computeHash(), 266 },
    { Amino::getTypeId<Bifrost::Math::bool2>().computeHash(), 267 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool2>>>().computeHash(), 268 },
    { Amino::getTypeId<Bifrost::Math::bool3>().computeHash(), 269 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool3>>>().computeHash(), 270 },
    { Amino::getTypeId<Bifrost::Math::bool4>().computeHash(), 271 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool4>>>().computeHash(), 272 },
    { Amino::getTypeId<Bifrost::Math::bool2x2>().computeHash(), 273 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool2x2>>>().computeHash(), 274 },
    { Amino::getTypeId<Bifrost::Math::bool2x3>().computeHash(), 275 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool2x3>>>().computeHash(), 276 },
    { Amino::getTypeId<Bifrost::Math::bool2x4>().computeHash(), 277 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool2x4>>>().computeHash(), 278 },
    { Amino::getTypeId<Bifrost::Math::bool3x2>().computeHash(), 279 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool3x2>>>().computeHash(), 280 },
    { Amino::getTypeId<Bifrost::Math::bool3x3>().computeHash(), 281 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool3x3>>>().computeHash(), 282 },
    { Amino::getTypeId<Bifrost::Math::bool3x4>().computeHash(), 283 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool3x4>>>().computeHash(), 284 },
    { Amino::getTypeId<Bifrost::Math::bool4x2>().computeHash(), 285 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool4x2>>>().computeHash(), 286 },
    { Amino::getTypeId<Bifrost::Math::bool4x3>().computeHash(), 287 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool4x3>>>().computeHash(), 288 },
    { Amino::getTypeId<Bifrost::Math::bool4x4>().computeHash(), 289 },
    { Amino::getTypeId<Amino::Ptr<Amino::Array<Bifrost::Math::bool4x4>>>().computeHash(), 290 }
};

int initToPython() {
    import_array();
    return 0;
}

// To Python functions =======================================================================================================================
namespace ToPython
{
    PyObject* fromSimple(Amino::Ptr<Bifrost::Object> bob) {
        PyObject* dict = PyDict_New();
        auto keys = bob->keys();
        for (int i = 0; i < keys->size(); i++) {
            const char* key = keys->at(i).c_str();
            auto py_obj = anyToPy(bob->getProperty(key));
            if (py_obj != nullptr) {
                PyDict_SetItemString(dict, key, py_obj);
                if (Py_REFCNT(py_obj) > 1)
                    Py_DECREF(py_obj);
			}
            else
                PyDict_SetItemString(dict, key, Py_None);            
        }
        return dict;
    }

    PyObject* fromSimple(Amino::String string) {
        PyObject* py_obj;
        py_obj = PyUnicode_FromString(string.c_str());
        return py_obj;
    }

    // These are no longer in use as now Numpy handles all numeric types and arrays, keeping incase it comes in handy.
    /*
    PyObject* fromSimple(float number) {
        return PyFloat_FromDouble(number);
    }

    PyObject* fromSimple(long long number) {
        return PyLong_FromLongLong(number);
    }

    PyObject* fromSimple(unsigned long long number) {
        return PyLong_FromLongLong(number);
    }

    PyObject* fromSimple(int number) {
        return PyLong_FromLong(number);
    }

    PyObject* fromSimple(unsigned int number) {
        return PyLong_FromLong(number);
    }

    PyObject* fromSimple(short number) {
        return PyLong_FromLong(number);
    }

    PyObject* fromSimple(unsigned short number) {
        return PyLong_FromLong(number);
    }

    PyObject* fromSimple(char number) {
        return PyLong_FromLong(number);
    }

    PyObject* fromSimple(unsigned char number) {
        return PyLong_FromLong(number);
    }

    PyObject* fromSimple(bool number) {
        if (number)
            return Py_True;
        return Py_False;
    }

    template <typename T>
    PyObject* fromVector2(T vector) {
        auto x = fromSimple(vector.x);
        auto y = fromSimple(vector.y);
        auto py_vector = PyTuple_Pack(2, x, y);
        Py_DECREF(x); Py_DECREF(y);

        return py_vector;
    }

    template <typename T>
    PyObject* fromVector3(T vector) {
        auto x = fromSimple(vector.x);
        auto y = fromSimple(vector.y);
        auto z = fromSimple(vector.x);
        auto py_vector = PyTuple_Pack(3, x, y, z);
        Py_DECREF(x); Py_DECREF(y); Py_DECREF(z);

        return py_vector;
    }

    template <typename T>
    PyObject* fromVector4(T vector) {
        auto x = fromSimple(vector.x);
        auto y = fromSimple(vector.y);
        auto z = fromSimple(vector.x);
        auto w = fromSimple(vector.w);
        auto py_vector = PyTuple_Pack(4, x, y, z, w);
        Py_DECREF(x); Py_DECREF(y); Py_DECREF(z); Py_DECREF(w);

        return py_vector;
    }

    template <typename T>
    PyObject* fromVectorX(T vector, int dims) {
        switch (dims) {
        case 2:
            return fromVector2(vector);
        case 3:
            return fromVector3(vector);
        case 4:
            return fromVector4(vector);
        }
    }

    template <typename T>
    PyObject* fromMatrixXx2(T matrix, int rows) {
        auto c0 = fromVectorX(matrix.c0, rows);
        auto c1 = fromVectorX(matrix.c1, rows);
        auto py_matrix = PyTuple_Pack(2, c0, c1);
        Py_DECREF(c0); Py_DECREF(c1);
        return py_matrix;

    }

    template <typename T>
    PyObject* fromMatrixXx3(T matrix, int rows) {
        auto c0 = fromVectorX(matrix.c0, rows);
        auto c1 = fromVectorX(matrix.c1, rows);
        auto c2 = fromVectorX(matrix.c2, rows);
        auto py_matrix = PyTuple_Pack(3, c0, c1, c2);
        Py_DECREF(c0); Py_DECREF(c1); Py_DECREF(c2);
        return py_matrix;

    }

    template <typename T>
    PyObject* fromMatrixXx4(T matrix, int rows) {
        auto c0 = fromVectorX(matrix.c0, rows);
        auto c1 = fromVectorX(matrix.c1, rows);
        auto c2 = fromVectorX(matrix.c2, rows);
        auto c3 = fromVectorX(matrix.c3, rows);
        auto py_matrix = PyTuple_Pack(4, c0, c1, c2, c3);
        Py_DECREF(c0); Py_DECREF(c1); Py_DECREF(c2); Py_DECREF(c3);
        return py_matrix;

    }

    template <typename T>
    PyObject* listFromSimpleArray(Amino::Ptr<Amino::Array<T>> amino_array) {
        PyObject* pyList = PyList_New((int)amino_array->size());
        for (int i = 0; i < amino_array->size(); i++) {
            auto val = fromSimple(amino_array->at(i));
            PyList_SetItem(pyList, i, val);
        }
        return pyList;
    }
    */
}

namespace ToNumpy
{
    void setTypeMetadata(PyArrayObject* np_array, const char* value) {
		PyObject* dict = PyDict_New();
        PyObject* py_value = PyUnicode_FromString(value);
        PyDict_SetItemString(dict, "bifrost_type", py_value);
		np_array->descr->metadata = dict;
        Py_DECREF(py_value); Py_DECREF(dict);
	}

    PyObject* fromScalar(Amino::Any scalar, int np_type) {
        void* data = const_cast<void*>(static_cast<const void*>(&scalar));
        auto py_obj = PyArray_Scalar(data, PyArray_DescrFromType(np_type), nullptr);
        return py_obj;
    }

    template <typename T>
    PyObject* fromVector(T vector, int np_type, int num_members) {
        npy_intp shape[1] = { num_members };
        auto amino_array = Amino::Array<T>(1);
        amino_array[0] = vector;
        void* data = const_cast<void*>(static_cast<const void*>(amino_array.data()));
        PyObject* np_array = PyArray_SimpleNew(1, shape, np_type);
        memcpy(PyArray_DATA(np_array), data, sizeof(T));
        setTypeMetadata(reinterpret_cast<PyArrayObject*>(np_array), "vector");
        return np_array;
    }

    template <typename T>
    PyObject* fromMatrix(T matrix, int np_type, int num_inner, int num_outer)
    {
        npy_intp shape[2] = { num_outer, num_inner };
        auto amino_array = Amino::Array<T>(1);
        amino_array[0] = matrix;
        void* data = const_cast<void*>(static_cast<const void*>(amino_array.data()));
        PyObject* np_array = PyArray_SimpleNew(2, shape, np_type);
        memcpy(PyArray_DATA(np_array), data, sizeof(T));
        setTypeMetadata(reinterpret_cast<PyArrayObject*>(np_array), "matrix");
        return np_array;
    }

    template <typename T>
    PyObject* fromArray(Amino::Ptr<Amino::Array<T>> amino_array, int np_type, int num_dims, npy_intp shape[]) {
        shape[0] = amino_array->size();

        //void* data = malloc(amino_array->size() * sizeof(T));
        //memcpy(data, amino_array->data(), amino_array->size() * sizeof(T));
        void* data = const_cast<void*>(static_cast<const void*>(amino_array->data()));
        PyObject* np_array = PyArray_SimpleNewFromData(num_dims, shape, np_type, data);
        return np_array;
    }

    template <typename T>
    PyObject* objectArrayfromArray(Amino::Ptr<Amino::Array<T>> amino_array, bool is_strings) {
        npy_intp shape[1] = { amino_array->size() };
        auto py_obj = PyArray_SimpleNew(1, shape, NPY_OBJECT);
        PyArrayObject* np_array = reinterpret_cast<PyArrayObject*>(py_obj);
        for (int i = 0; i < amino_array->size(); i++) {
			auto item_py_obj = ToPython::fromSimple(amino_array->at(i));
			PyArray_SETITEM(np_array, PyArray_GETPTR1(np_array, i), item_py_obj);
            if (Py_REFCNT(item_py_obj) > 1)
                Py_DECREF(item_py_obj);
		}

        if (is_strings) {
			setTypeMetadata(np_array, "string_array");
		}

		return py_obj;
	}
}


PyObject* anyToPy(Amino::Any data) {
    auto type_hash = data.type().computeHash();
    npy_intp shape[3];
    PyObject* py_obj;

    switch (TYPE_MAP[type_hash]) {
    case 1:
        return ToPython::fromSimple(Amino::any_cast<Amino::Ptr<Bifrost::Object>>(data));
    case 2:
        return ToNumpy::objectArrayfromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Amino::Ptr<Bifrost::Object>>>>(data), false);
    case 3:
        return ToPython::fromSimple(Amino::any_cast<Amino::String>(data));
    case 4:
        return ToNumpy::objectArrayfromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Amino::String>>>(data), true);
    case 5:
        return ToNumpy::fromScalar(data, NPY_FLOAT);
    case 6:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<float>>>(data), NPY_FLOAT, 1, shape);
    case 7:
        return ToNumpy::fromScalar(data, NPY_DOUBLE);
    case 8:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<double>>>(data), NPY_DOUBLE, 1, shape);
    case 9:
        return ToNumpy::fromScalar(data, NPY_LONGLONG);
    case 10:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<long long>>>(data), NPY_LONGLONG, 1, shape);
    case 11:
        return ToNumpy::fromScalar(data, NPY_ULONGLONG);
    case 12:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<ulong>>>(data), NPY_ULONGLONG, 1, shape);
    case 13:
        return ToNumpy::fromScalar(data, NPY_INT);
    case 14:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<int>>>(data), NPY_INT, 1, shape);
    case 15:
        return ToNumpy::fromScalar(data, NPY_UINT);
    case 16:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<uint>>>(data), NPY_UINT, 1, shape);
    case 17:
        return ToNumpy::fromScalar(data, NPY_SHORT);
    case 18:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<short>>>(data), NPY_SHORT, 1, shape);
    case 19:
        return ToNumpy::fromScalar(data, NPY_USHORT);
    case 20:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<ushort>>>(data), NPY_USHORT, 1, shape);
    case 21:
        return ToNumpy::fromScalar(data, NPY_BYTE);
    case 22:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<signed char>>>(data), NPY_BYTE, 1, shape);
    case 23:
        return ToNumpy::fromScalar(data, NPY_UBYTE);
    case 24:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<uchar>>>(data), NPY_UBYTE, 1, shape);
    case 25:
        return ToNumpy::fromScalar(data, NPY_BOOL);
    case 26:
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<bool>>>(data), NPY_BOOL, 1, shape);
    case 27:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::float2>(data), NPY_FLOAT, 2);
    case 28:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float2>>>(data), NPY_FLOAT, 2, shape);
    case 29:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::float3>(data), NPY_FLOAT, 3);
    case 30:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float3>>>(data), NPY_FLOAT, 2, shape);
    case 31:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::float4>(data), NPY_FLOAT, 4);
    case 32:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float4>>>(data), NPY_FLOAT, 2, shape);
    case 33:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float2x2>(data), NPY_FLOAT, 2, 2);
    case 34:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float2x2>>>(data), NPY_FLOAT, 3, shape);
    case 35:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float2x3>(data), NPY_FLOAT, 2, 3);
    case 36:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float2x3>>>(data), NPY_FLOAT, 3, shape);
    case 37:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float2x4>(data), NPY_FLOAT, 2, 4);
    case 38:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float2x4>>>(data), NPY_FLOAT, 3, shape);
    case 39:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float3x2>(data), NPY_FLOAT, 3, 2);
    case 40:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float3x2>>>(data), NPY_FLOAT, 3, shape);
    case 41:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float3x3>(data), NPY_FLOAT, 3, 3);
    case 42:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float3x3>>>(data), NPY_FLOAT, 3, shape);
    case 43:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float3x4>(data), NPY_FLOAT, 3, 4);
    case 44:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float3x4>>>(data), NPY_FLOAT, 3, shape);
    case 45:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float4x2>(data), NPY_FLOAT, 4, 2);
    case 46:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float4x2>>>(data), NPY_FLOAT, 3, shape);
    case 47:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float4x3>(data), NPY_FLOAT, 4, 3);
    case 48:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float4x3>>>(data), NPY_FLOAT, 3, shape);
    case 49:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::float4x4>(data), NPY_FLOAT, 4, 4);
    case 50:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::float4x4>>>(data), NPY_FLOAT, 3, shape);
    case 51:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::double2>(data), NPY_DOUBLE, 2);
    case 52:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double2>>>(data), NPY_DOUBLE, 2, shape);
    case 53:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::double3>(data), NPY_DOUBLE, 3);
    case 54:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double3>>>(data), NPY_DOUBLE, 2, shape);
    case 55:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::double4>(data), NPY_DOUBLE, 4);
    case 56:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double4>>>(data), NPY_DOUBLE, 2, shape);
    case 57:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double2x2>(data), NPY_DOUBLE, 2, 2);
    case 58:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double2x2>>>(data), NPY_DOUBLE, 3, shape);
    case 59:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double2x3>(data), NPY_DOUBLE, 2, 3);
    case 60:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double2x3>>>(data), NPY_DOUBLE, 3, shape);
    case 61:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double2x4>(data), NPY_DOUBLE, 2, 4);
    case 62:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double2x4>>>(data), NPY_DOUBLE, 3, shape);
    case 63:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double3x2>(data), NPY_DOUBLE, 3, 2);
    case 64:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double3x2>>>(data), NPY_DOUBLE, 3, shape);
    case 65:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double3x3>(data), NPY_DOUBLE, 3, 3);
    case 66:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double3x3>>>(data), NPY_DOUBLE, 3, shape);
    case 67:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double3x4>(data), NPY_DOUBLE, 3, 4);
    case 68:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double3x4>>>(data), NPY_DOUBLE, 3, shape);
    case 69:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double4x2>(data), NPY_DOUBLE, 4, 2);
    case 70:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double4x2>>>(data), NPY_DOUBLE, 3, shape);
    case 71:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double4x3>(data), NPY_DOUBLE, 4, 3);
    case 72:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double4x3>>>(data), NPY_DOUBLE, 3, shape);
    case 73:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::double4x4>(data), NPY_DOUBLE, 4, 4);
    case 74:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::double4x4>>>(data), NPY_DOUBLE, 3, shape);
    case 75:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::long2>(data), NPY_LONGLONG, 2);
    case 76:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long2>>>(data), NPY_LONGLONG, 2, shape);
    case 77:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::long3>(data), NPY_LONGLONG, 3);
    case 78:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long3>>>(data), NPY_LONGLONG, 2, shape);
    case 79:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::long4>(data), NPY_LONGLONG, 4);
    case 80:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long4>>>(data), NPY_LONGLONG, 2, shape);
    case 81:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long2x2>(data), NPY_LONGLONG, 2, 2);
    case 82:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long2x2>>>(data), NPY_LONGLONG, 3, shape);
    case 83:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long2x3>(data), NPY_LONGLONG, 2, 3);
    case 84:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long2x3>>>(data), NPY_LONGLONG, 3, shape);
    case 85:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long2x4>(data), NPY_LONGLONG, 2, 4);
    case 86:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long2x4>>>(data), NPY_LONGLONG, 3, shape);
    case 87:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long3x2>(data), NPY_LONGLONG, 3, 2);
    case 88:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long3x2>>>(data), NPY_LONGLONG, 3, shape);
    case 89:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long3x3>(data), NPY_LONGLONG, 3, 3);
    case 90:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long3x3>>>(data), NPY_LONGLONG, 3, shape);
    case 91:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long3x4>(data), NPY_LONGLONG, 3, 4);
    case 92:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long3x4>>>(data), NPY_LONGLONG, 3, shape);
    case 93:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long4x2>(data), NPY_LONGLONG, 4, 2);
    case 94:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long4x2>>>(data), NPY_LONGLONG, 3, shape);
    case 95:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long4x3>(data), NPY_LONGLONG, 4, 3);
    case 96:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long4x3>>>(data), NPY_LONGLONG, 3, shape);
    case 97:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::long4x4>(data), NPY_LONGLONG, 4, 4);
    case 98:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::long4x4>>>(data), NPY_LONGLONG, 3, shape);
    case 99:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::ulong2>(data), NPY_ULONGLONG, 2);
    case 100:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2>>>(data), NPY_ULONGLONG, 2, shape);
    case 101:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::ulong3>(data), NPY_ULONGLONG, 3);
    case 102:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3>>>(data), NPY_ULONGLONG, 2, shape);
    case 103:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::ulong4>(data), NPY_ULONGLONG, 4);
    case 104:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4>>>(data), NPY_ULONGLONG, 2, shape);
    case 105:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong2x2>(data), NPY_ULONGLONG, 2, 2);
    case 106:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2x2>>>(data), NPY_ULONGLONG, 3, shape);
    case 107:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong2x3>(data), NPY_ULONGLONG, 2, 3);
    case 108:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2x3>>>(data), NPY_ULONGLONG, 3, shape);
    case 109:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong2x4>(data), NPY_ULONGLONG, 2, 4);
    case 110:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong2x4>>>(data), NPY_ULONGLONG, 3, shape);
    case 111:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong3x2>(data), NPY_ULONGLONG, 3, 2);
    case 112:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3x2>>>(data), NPY_ULONGLONG, 3, shape);
    case 113:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong3x3>(data), NPY_ULONGLONG, 3, 3);
    case 114:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3x3>>>(data), NPY_ULONGLONG, 3, shape);
    case 115:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong3x4>(data), NPY_ULONGLONG, 3, 4);
    case 116:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong3x4>>>(data), NPY_ULONGLONG, 3, shape);
    case 117:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong4x2>(data), NPY_ULONGLONG, 4, 2);
    case 118:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4x2>>>(data), NPY_ULONGLONG, 3, shape);
    case 119:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong4x3>(data), NPY_ULONGLONG, 4, 3);
    case 120:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4x3>>>(data), NPY_ULONGLONG, 3, shape);
    case 121:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ulong4x4>(data), NPY_ULONGLONG, 4, 4);
    case 122:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ulong4x4>>>(data), NPY_ULONGLONG, 3, shape);
    case 123:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::int2>(data), NPY_INT, 2);
    case 124:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int2>>>(data), NPY_INT, 2, shape);
    case 125:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::int3>(data), NPY_INT, 3);
    case 126:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int3>>>(data), NPY_INT, 2, shape);
    case 127:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::int4>(data), NPY_INT, 4);
    case 128:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int4>>>(data), NPY_INT, 2, shape);
    case 129:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int2x2>(data), NPY_INT, 2, 2);
    case 130:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int2x2>>>(data), NPY_INT, 3, shape);
    case 131:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int2x3>(data), NPY_INT, 2, 3);
    case 132:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int2x3>>>(data), NPY_INT, 3, shape);
    case 133:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int2x4>(data), NPY_INT, 2, 4);
    case 134:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int2x4>>>(data), NPY_INT, 3, shape);
    case 135:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int3x2>(data), NPY_INT, 3, 2);
    case 136:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int3x2>>>(data), NPY_INT, 3, shape);
    case 137:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int3x3>(data), NPY_INT, 3, 3);
    case 138:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int3x3>>>(data), NPY_INT, 3, shape);
    case 139:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int3x4>(data), NPY_INT, 3, 4);
    case 140:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int3x4>>>(data), NPY_INT, 3, shape);
    case 141:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int4x2>(data), NPY_INT, 4, 2);
    case 142:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int4x2>>>(data), NPY_INT, 3, shape);
    case 143:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int4x3>(data), NPY_INT, 4, 3);
    case 144:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int4x3>>>(data), NPY_INT, 3, shape);
    case 145:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::int4x4>(data), NPY_INT, 4, 4);
    case 146:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::int4x4>>>(data), NPY_INT, 3, shape);
    case 147:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::uint2>(data), NPY_UINT, 2);
    case 148:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint2>>>(data), NPY_UINT, 2, shape);
    case 149:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::uint3>(data), NPY_UINT, 3);
    case 150:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint3>>>(data), NPY_UINT, 2, shape);
    case 151:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::uint4>(data), NPY_UINT, 4);
    case 152:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint4>>>(data), NPY_UINT, 2, shape);
    case 153:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint2x2>(data), NPY_UINT, 2, 2);
    case 154:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint2x2>>>(data), NPY_UINT, 3, shape);
    case 155:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint2x3>(data), NPY_UINT, 2, 3);
    case 156:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint2x3>>>(data), NPY_UINT, 3, shape);
    case 157:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint2x4>(data), NPY_UINT, 2, 4);
    case 158:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint2x4>>>(data), NPY_UINT, 3, shape);
    case 159:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint3x2>(data), NPY_UINT, 3, 2);
    case 160:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint3x2>>>(data), NPY_UINT, 3, shape);
    case 161:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint3x3>(data), NPY_UINT, 3, 3);
    case 162:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint3x3>>>(data), NPY_UINT, 3, shape);
    case 163:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint3x4>(data), NPY_UINT, 3, 4);
    case 164:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint3x4>>>(data), NPY_UINT, 3, shape);
    case 165:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint4x2>(data), NPY_UINT, 4, 2);
    case 166:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint4x2>>>(data), NPY_UINT, 3, shape);
    case 167:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint4x3>(data), NPY_UINT, 4, 3);
    case 168:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint4x3>>>(data), NPY_UINT, 3, shape);
    case 169:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uint4x4>(data), NPY_UINT, 4, 4);
    case 170:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uint4x4>>>(data), NPY_UINT, 3, shape);
    case 171:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::short2>(data), NPY_SHORT, 2);
    case 172:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short2>>>(data), NPY_SHORT, 2, shape);
    case 173:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::short3>(data), NPY_SHORT, 3);
    case 174:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short3>>>(data), NPY_SHORT, 2, shape);
    case 175:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::short4>(data), NPY_SHORT, 4);
    case 176:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short4>>>(data), NPY_SHORT, 2, shape);
    case 177:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short2x2>(data), NPY_SHORT, 2, 2);
    case 178:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short2x2>>>(data), NPY_SHORT, 3, shape);
    case 179:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short2x3>(data), NPY_SHORT, 2, 3);
    case 180:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short2x3>>>(data), NPY_SHORT, 3, shape);
    case 181:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short2x4>(data), NPY_SHORT, 2, 4);
    case 182:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short2x4>>>(data), NPY_SHORT, 3, shape);
    case 183:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short3x2>(data), NPY_SHORT, 3, 2);
    case 184:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short3x2>>>(data), NPY_SHORT, 3, shape);
    case 185:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short3x3>(data), NPY_SHORT, 3, 3);
    case 186:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short3x3>>>(data), NPY_SHORT, 3, shape);
    case 187:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short3x4>(data), NPY_SHORT, 3, 4);
    case 188:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short3x4>>>(data), NPY_SHORT, 3, shape);
    case 189:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short4x2>(data), NPY_SHORT, 4, 2);
    case 190:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short4x2>>>(data), NPY_SHORT, 3, shape);
    case 191:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short4x3>(data), NPY_SHORT, 4, 3);
    case 192:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short4x3>>>(data), NPY_SHORT, 3, shape);
    case 193:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::short4x4>(data), NPY_SHORT, 4, 4);
    case 194:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::short4x4>>>(data), NPY_SHORT, 3, shape);
    case 195:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::ushort2>(data), NPY_USHORT, 2);
    case 196:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2>>>(data), NPY_USHORT, 2, shape);
    case 197:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::ushort3>(data), NPY_USHORT, 3);
    case 198:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3>>>(data), NPY_USHORT, 2, shape);
    case 199:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::ushort4>(data), NPY_USHORT, 4);
    case 200:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4>>>(data), NPY_USHORT, 2, shape);
    case 201:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort2x2>(data), NPY_USHORT, 2, 2);
    case 202:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2x2>>>(data), NPY_USHORT, 3, shape);
    case 203:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort2x3>(data), NPY_USHORT, 2, 3);
    case 204:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2x3>>>(data), NPY_USHORT, 3, shape);
    case 205:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort2x4>(data), NPY_USHORT, 2, 4);
    case 206:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort2x4>>>(data), NPY_USHORT, 3, shape);
    case 207:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort3x2>(data), NPY_USHORT, 3, 2);
    case 208:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3x2>>>(data), NPY_USHORT, 3, shape);
    case 209:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort3x3>(data), NPY_USHORT, 3, 3);
    case 210:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3x3>>>(data), NPY_USHORT, 3, shape);
    case 211:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort3x4>(data), NPY_USHORT, 3, 4);
    case 212:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort3x4>>>(data), NPY_USHORT, 3, shape);
    case 213:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort4x2>(data), NPY_USHORT, 4, 2);
    case 214:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4x2>>>(data), NPY_USHORT, 3, shape);
    case 215:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort4x3>(data), NPY_USHORT, 4, 3);
    case 216:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4x3>>>(data), NPY_USHORT, 3, shape);
    case 217:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::ushort4x4>(data), NPY_USHORT, 4, 4);
    case 218:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::ushort4x4>>>(data), NPY_USHORT, 3, shape);
    case 219:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::char2>(data), NPY_BYTE, 2);
    case 220:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char2>>>(data), NPY_BYTE, 2, shape);
    case 221:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::char3>(data), NPY_BYTE, 3);
    case 222:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char3>>>(data), NPY_BYTE, 2, shape);
    case 223:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::char4>(data), NPY_BYTE, 4);
    case 224:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char4>>>(data), NPY_BYTE, 2, shape);
    case 225:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char2x2>(data), NPY_BYTE, 2, 2);
    case 226:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char2x2>>>(data), NPY_BYTE, 3, shape);
    case 227:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char2x3>(data), NPY_BYTE, 2, 3);
    case 228:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char2x3>>>(data), NPY_BYTE, 3, shape);
    case 229:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char2x4>(data), NPY_BYTE, 2, 4);
    case 230:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char2x4>>>(data), NPY_BYTE, 3, shape);
    case 231:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char3x2>(data), NPY_BYTE, 3, 2);
    case 232:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char3x2>>>(data), NPY_BYTE, 3, shape);
    case 233:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char3x3>(data), NPY_BYTE, 3, 3);
    case 234:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char3x3>>>(data), NPY_BYTE, 3, shape);
    case 235:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char3x4>(data), NPY_BYTE, 3, 4);
    case 236:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char3x4>>>(data), NPY_BYTE, 3, shape);
    case 237:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char4x2>(data), NPY_BYTE, 4, 2);
    case 238:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char4x2>>>(data), NPY_BYTE, 3, shape);
    case 239:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char4x3>(data), NPY_BYTE, 4, 3);
    case 240:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char4x3>>>(data), NPY_BYTE, 3, shape);
    case 241:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::char4x4>(data), NPY_BYTE, 4, 4);
    case 242:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::char4x4>>>(data), NPY_BYTE, 3, shape);
    case 243:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::uchar2>(data), NPY_UBYTE, 2);
    case 244:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2>>>(data), NPY_UBYTE, 2, shape);
    case 245:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::uchar3>(data), NPY_UBYTE, 3);
    case 246:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3>>>(data), NPY_UBYTE, 2, shape);
    case 247:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::uchar4>(data), NPY_UBYTE, 4);
    case 248:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4>>>(data), NPY_UBYTE, 2, shape);
    case 249:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar2x2>(data), NPY_UBYTE, 2, 2);
    case 250:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2x2>>>(data), NPY_UBYTE, 3, shape);
    case 251:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar2x3>(data), NPY_UBYTE, 2, 3);
    case 252:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2x3>>>(data), NPY_UBYTE, 3, shape);
    case 253:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar2x4>(data), NPY_UBYTE, 2, 4);
    case 254:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar2x4>>>(data), NPY_UBYTE, 3, shape);
    case 255:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar3x2>(data), NPY_UBYTE, 3, 2);
    case 256:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3x2>>>(data), NPY_UBYTE, 3, shape);
    case 257:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar3x3>(data), NPY_UBYTE, 3, 3);
    case 258:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3x3>>>(data), NPY_UBYTE, 3, shape);
    case 259:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar3x4>(data), NPY_UBYTE, 3, 4);
    case 260:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar3x4>>>(data), NPY_UBYTE, 3, shape);
    case 261:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar4x2>(data), NPY_UBYTE, 4, 2);
    case 262:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4x2>>>(data), NPY_UBYTE, 3, shape);
    case 263:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar4x3>(data), NPY_UBYTE, 4, 3);
    case 264:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4x3>>>(data), NPY_UBYTE, 3, shape);
    case 265:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::uchar4x4>(data), NPY_UBYTE, 4, 4);
    case 266:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::uchar4x4>>>(data), NPY_UBYTE, 3, shape);
    case 267:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::bool2>(data), NPY_BOOL, 2);
    case 268:
        shape[1] = 2;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool2>>>(data), NPY_BOOL, 2, shape);
    case 269:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::bool3>(data), NPY_BOOL, 3);
    case 270:
        shape[1] = 3;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool3>>>(data), NPY_BOOL, 2, shape);
    case 271:
        return ToNumpy::fromVector(Amino::any_cast<Bifrost::Math::bool4>(data), NPY_BOOL, 4);
    case 272:
        shape[1] = 4;
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool4>>>(data), NPY_BOOL, 2, shape);
    case 273:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool2x2>(data), NPY_BOOL, 2, 2);
    case 274:
        shape[1] = 2; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool2x2>>>(data), NPY_BOOL, 3, shape);
    case 275:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool2x3>(data), NPY_BOOL, 2, 3);
    case 276:
        shape[1] = 3; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool2x3>>>(data), NPY_BOOL, 3, shape);
    case 277:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool2x4>(data), NPY_BOOL, 2, 4);
    case 278:
        shape[1] = 4; shape[2] = { 2 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool2x4>>>(data), NPY_BOOL, 3, shape);
    case 279:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool3x2>(data), NPY_BOOL, 3, 2);
    case 280:
        shape[1] = 2; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool3x2>>>(data), NPY_BOOL, 3, shape);
    case 281:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool3x3>(data), NPY_BOOL, 3, 3);
    case 282:
        shape[1] = 3; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool3x3>>>(data), NPY_BOOL, 3, shape);
    case 283:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool3x4>(data), NPY_BOOL, 3, 4);
    case 284:
        shape[1] = 4; shape[2] = { 3 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool3x4>>>(data), NPY_BOOL, 3, shape);
    case 285:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool4x2>(data), NPY_BOOL, 4, 2);
    case 286:
        shape[1] = 2; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool4x2>>>(data), NPY_BOOL, 3, shape);
    case 287:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool4x3>(data), NPY_BOOL, 4, 3);
    case 288:
        shape[1] = 3; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool4x3>>>(data), NPY_BOOL, 3, shape);
    case 289:
        return ToNumpy::fromMatrix(Amino::any_cast<Bifrost::Math::bool4x4>(data), NPY_BOOL, 4, 4);
    case 290:
        shape[1] = 4; shape[2] = { 4 };
        return ToNumpy::fromArray(Amino::any_cast<Amino::Ptr<Amino::Array<Bifrost::Math::bool4x4>>>(data), NPY_BOOL, 3, shape);
    default:
        return nullptr;
	}
}