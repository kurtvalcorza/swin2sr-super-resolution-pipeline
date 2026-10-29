"""Paired low-/high-resolution datasets for the adaptation contract: the digest-pinned iNaturalist photograph
sample, the degradation that turns a high-resolution crop into its low-resolution input, the record contract and
its structural validation, photograph-disjoint splitting, and the BYOD loader.

A record is ``{id, lr, hr}`` where ``hr`` is a PIL image (RGB, sides at most ``MAX_HR_SIDE``, multiples of
``2 * WINDOW``) and ``lr`` is a PIL image of exactly half its size — the pair the model is asked to reverse. An
optional ``category`` (free text, at most 32 characters) is carried into the per-category breakdown.

The default sample is 360 research-grade, CC0-licensed iNaturalist photographs of six North American bird species
(the same pinned photographs the fleet's SigLIP 2 tutorial uses), each pinned by photo id, byte size and SHA-256 and
fetched from the iNaturalist open-data bucket at run time. The high-resolution image is a centred ``HR_CROP``-pixel
crop of the served photo; the low-resolution input is that crop downsampled 2x with a bicubic filter and then
JPEG-compressed at ``JPEG_QUALITY`` (``degrade``) — a degradation the classical-SR checkpoint was not trained for,
which is what the adaptation is asked to learn. The species is the record's ``category``.

Review R88-M4 (owner ruling 2026-10-17): every one of the 360 photographs has been trained on, used for selection or
scored, so the seed-42 test split (96 crops) is now the **exposed development test**, and a **fresh test** of 144
further photographs (24 per species) is pinned in ``FRESH_TEST_RECORDS``: CC0, research grade, one per observer,
observers disjoint from the 360 and from every other fleet pin, drawn by the seeded, documented query in
``tools/draw_fresh_test.py`` (record: ``docs/fresh-test-draw.json``). The fresh test is scored only under the
pre-registered settings (``PREREGISTERED_SETTINGS``; ``docs/swin2sr-preregistration.md``).
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MAX_INPUT_SIDE, MIN_INPUT_SIDE, MODEL_ID, MODEL_REVISION, UPSCALE

CORPUS_NAME = "iNaturalist CC0 bird photographs (six species)"
CORPUS_RELEASE = (
    "iNaturalist open data: research-grade observations whose photos carry the CC0 1.0 licence, one photo per "
    "observer per species, fetched from the open-data bucket by photo id"
)
CORPUS_BASE_URL = "https://inaturalist-open-data.s3.amazonaws.com/photos/"
CORPUS_LICENSE = (
    "CC0 1.0 (each photo's licence as published by its observer on iNaturalist; attribution is kept per record)"
)
CORPUS_BYTES = 39_223_447
DEFAULT_CACHE_DIR = Path("weights") / "inat-birds"

HR_CROP = 192  # px; a multiple of 2 * WINDOW so the LR input (96 px) needs no padding
WINDOW = 8  # config.json "window_size"
JPEG_QUALITY = 40  # the LR degradation beyond bicubic downsampling
MAX_HR_SIDE = MAX_INPUT_SIDE * UPSCALE
MIN_HR_SIDE = MIN_INPUT_SIDE * UPSCALE
MAX_IMAGE_SIDE = 4096  # served photos larger than this are rejected before any crop
SAMPLE_SPLIT = {"train": 36, "validation": 8, "test": 16}  # per species; 6 species -> 216 / 48 / 96
SAMPLE_SEED = 42
SAMPLE_DIGEST = "827eb4a31619d7073a825ce5e5a46b58f26e07b3c920d6cf75b6f11a1241ca83"  # dataset_digest over the three default splits together; tests pin it
MIN_RECORDS = 8
MAX_RECORDS = 5_000
EXPOSED_TEST_LABEL = "exposed development test"  # the seed-42 test split: looked at during development (R88-M1/M4)
FRESH_TEST_LABEL = "fresh test"
FRESH_TEST_PER_SPECIES = 24
MAX_CATEGORY_CHARS = 32
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

SPECIES: dict[str, tuple[str, str]] = {
    "song_sparrow": ("Melospiza melodia", "Song Sparrow"),
    "chipping_sparrow": ("Spizella passerina", "Chipping Sparrow"),
    "white_throated_sparrow": ("Zonotrichia albicollis", "White-throated Sparrow"),
    "dark_eyed_junco": ("Junco hyemalis", "Dark-eyed Junco"),
    "house_finch": ("Haemorhous mexicanus", "House Finch"),
    "american_goldfinch": ("Spinus tristis", "American Goldfinch"),
}
# (id, label, iNat photo id, iNat observation id, observer login, bytes, sha256 of the served
#  <photo id>/medium.<ext>, ext) — the bucket serves each photo under its original extension
#  (jpg or jpeg); the digest pins the served bytes
SAMPLE_RECORDS: tuple[tuple[str, str, int, int, str, int, str, str], ...] = (
    (
        "song_sparrow-00",
        "song_sparrow",
        129376982,
        79016324,
        "andywilson",
        43427,
        "7a9d9304a82f202e992655ec5f65477cd3d7c1dce03aa89a214c2daa38f9d61d",
        "jpg",
    ),
    (
        "song_sparrow-01",
        "song_sparrow",
        480991086,
        267636534,
        "lyneisfilm",
        162073,
        "11f77ff277dd2703c1000f2c787136ff0c3ca7ffad7017056892fe789d65efec",
        "jpg",
    ),
    (
        "song_sparrow-02",
        "song_sparrow",
        546060381,
        302980489,
        "swpollinators",
        27899,
        "4e70b9519c6e5f7384a4495b91b45465f2b1599f86491d8f9f9b12635a4046f6",
        "jpg",
    ),
    (
        "song_sparrow-03",
        "song_sparrow",
        308625896,
        177450028,
        "radrat",
        70961,
        "1211da4fdb24ae85ef0c6c3e2d03542c430457856aec661fe8f5f2de0027eee5",
        "jpeg",
    ),
    (
        "song_sparrow-04",
        "song_sparrow",
        494793016,
        275349085,
        "k-simpkins",
        58410,
        "255538cf450197257e86ed3d41dc69fb78e594434e9cb338c6314288c6cff26e",
        "jpg",
    ),
    (
        "song_sparrow-05",
        "song_sparrow",
        674054489,
        369029444,
        "ben142",
        220573,
        "1ae24622888d9d449ffd6b5c65cac1b9b14870fd8e12dbed8aa0acc2f5030123",
        "jpg",
    ),
    (
        "song_sparrow-06",
        "song_sparrow",
        339623726,
        193339933,
        "rawcomposition",
        25012,
        "d2cde085277a71886a2bf211a1eec26941a73752375708726a8af29aa4995a8b",
        "jpg",
    ),
    (
        "song_sparrow-07",
        "song_sparrow",
        181658744,
        107953669,
        "gcart043",
        98482,
        "a662a6abb24f42b256fb6e2d6f02c3e128a1053ef534f454461aa5cadcc03fe4",
        "jpeg",
    ),
    (
        "song_sparrow-08",
        "song_sparrow",
        222768957,
        130949329,
        "davidfbird",
        110773,
        "0feee62753f409d0aa365e9aa017ddef436ad70a2673dc847feb3b5386af27ff",
        "jpg",
    ),
    (
        "song_sparrow-09",
        "song_sparrow",
        148994242,
        90171417,
        "glennberry",
        102702,
        "64333977d24957d723a1d26886004f222d810557617685579f72a6b553e508fa",
        "jpg",
    ),
    (
        "song_sparrow-10",
        "song_sparrow",
        637315932,
        349374463,
        "sooji",
        136572,
        "a5cebbc0cc2325d3805c4ac854e103f7ba1c22e3a34fb204f30d58681e865033",
        "jpg",
    ),
    (
        "song_sparrow-11",
        "song_sparrow",
        471146686,
        262252507,
        "jeanpaulboerekamps",
        98601,
        "4301f06b52b8dcf1e137567c32412e384edb71cb90e2e35459a0405b2e56529b",
        "jpg",
    ),
    (
        "song_sparrow-12",
        "song_sparrow",
        640811426,
        351179648,
        "erikschiff",
        107695,
        "27aecce184a485ee888c0e8101cb4a34899b8a185b62dc064f3dc2ec9902182b",
        "jpg",
    ),
    (
        "song_sparrow-13",
        "song_sparrow",
        123859010,
        75689904,
        "w_mark_c",
        190819,
        "6b6057a1c50b83ffeb4d9e34367b3e6a9b355236dbcae9820b509e1484f8fe33",
        "jpg",
    ),
    (
        "song_sparrow-14",
        "song_sparrow",
        108520869,
        67204020,
        "dugald",
        52496,
        "e483364889fb95c540db84b5edf5a2400febf7d2eb62a1e49303b13ad329d1b8",
        "jpg",
    ),
    (
        "song_sparrow-15",
        "song_sparrow",
        63537800,
        40010230,
        "nathanael15",
        51441,
        "5ae55f868e779a4e8ee34f6ad077b40c41f20aa699d967f373ebd659a89137a8",
        "jpg",
    ),
    (
        "song_sparrow-16",
        "song_sparrow",
        435251292,
        244042351,
        "carterdorscht",
        166662,
        "d0cdd9ddcf7202a91ac2c47910a0c230639bce50283e8511fb8311151763233a",
        "jpeg",
    ),
    (
        "song_sparrow-17",
        "song_sparrow",
        120710033,
        73898230,
        "tys_rbg",
        126820,
        "515b3b32b64e88d401990bfd1d8e2c1281443b5d1e763a09e86d3d3664e1df41",
        "jpg",
    ),
    (
        "song_sparrow-18",
        "song_sparrow",
        349361283,
        198243065,
        "sean579",
        42820,
        "5eb0031a8d6066650c66b265fb1724413273e095e4e531054d2817eb75910074",
        "jpg",
    ),
    (
        "song_sparrow-19",
        "song_sparrow",
        393408927,
        222067370,
        "irenemacaulay_",
        101681,
        "22e326807de963352b4307790bd40c5506adb0fab1f9844968edaf4bc5665e1c",
        "jpg",
    ),
    (
        "song_sparrow-20",
        "song_sparrow",
        614258628,
        337847585,
        "joy4birds",
        70552,
        "aa767aa5a74a76cfd985aba5282e589f670a9ce0a8fcdc6a096c0357992dec67",
        "jpg",
    ),
    (
        "song_sparrow-21",
        "song_sparrow",
        608802621,
        335154467,
        "jamesadney",
        112564,
        "6e94bce1b5135f48b0b19b64e21a0c76cff4f9b9a28fcbc2b7f5c8ddda6ef513",
        "jpg",
    ),
    (
        "song_sparrow-22",
        "song_sparrow",
        634100352,
        347744524,
        "zorthesosen",
        214044,
        "6a70dba26dfb3e98a244a9ec9badb812ddaa8b3612e7a36f66aba057bb757ecd",
        "jpg",
    ),
    (
        "song_sparrow-23",
        "song_sparrow",
        50065474,
        31954532,
        "truthseqr",
        82521,
        "4186fbf344e92038358d4338102aa440098bff5f558ab1d197f72fefbcec0b15",
        "jpeg",
    ),
    (
        "song_sparrow-24",
        "song_sparrow",
        8656044,
        6803564,
        "glmory",
        297964,
        "e8cab773436ccfaad4699e5112ef4edb516236d413d6dbf034eea7bf9c88c8f5",
        "jpg",
    ),
    (
        "song_sparrow-25",
        "song_sparrow",
        131051149,
        79988590,
        "funvill",
        72572,
        "0a694b5bb03aee6eb5a3a6132e855d47172b21b84cc4cc7c64367d1a31de7894",
        "jpeg",
    ),
    (
        "song_sparrow-26",
        "song_sparrow",
        12923156,
        9491600,
        "gambolingquail",
        69559,
        "6a92bff4fc76820c6f21e15c5f254385dd976a32264911bfc08417a1c2bf053d",
        "jpg",
    ),
    (
        "song_sparrow-27",
        "song_sparrow",
        12077500,
        8959545,
        "reuvenm",
        74010,
        "3e3c5e94c839f45610ef3ef7ffaf3575f4bfe365fc94454c30bfdb718ca3c593",
        "jpeg",
    ),
    (
        "song_sparrow-28",
        "song_sparrow",
        132730299,
        80955309,
        "steph123456",
        154309,
        "5f8854dd231a302c643b22521b49ed3007203d74b8b8413a23abc79ffe0b0270",
        "jpeg",
    ),
    (
        "song_sparrow-29",
        "song_sparrow",
        124840110,
        76316566,
        "terrimewbornagain",
        81046,
        "2020092b0b67397a78cc2b0df267fb671b3ba18e03fb0e35e664e0784061e3c6",
        "jpeg",
    ),
    (
        "song_sparrow-30",
        "song_sparrow",
        130188079,
        79482300,
        "fake_id",
        180462,
        "1d9ad7b45536b85d46fa0fd9a29d4a8ae828bc223836796a3eb9df1fe8fe4fc4",
        "jpeg",
    ),
    (
        "song_sparrow-31",
        "song_sparrow",
        136405967,
        83075831,
        "tom_lazar",
        178554,
        "f86c3c5899d20ac8fb73afb76d45666c610245636d7285a6d517b0081b9d6cc9",
        "jpeg",
    ),
    (
        "song_sparrow-32",
        "song_sparrow",
        118885385,
        72867352,
        "ellyne",
        105156,
        "aab53fb1e7c12adc697711215778c81983f51af83e20bfffe841b80c1d144673",
        "jpeg",
    ),
    (
        "song_sparrow-33",
        "song_sparrow",
        676467457,
        370293984,
        "sholsenbeck",
        59104,
        "b7c230ca942cedc1c53f42902e1f70bc2d09497e57d2e9f31dfe72a5f6c84d16",
        "jpg",
    ),
    (
        "song_sparrow-34",
        "song_sparrow",
        688316087,
        376461614,
        "doublecritch",
        109416,
        "7f11c69df733f99d6e8663e91d807327f8cc57f404049fdbe95e2fd1ce1066e1",
        "jpg",
    ),
    (
        "song_sparrow-35",
        "song_sparrow",
        691785499,
        378254663,
        "karuquebec",
        144522,
        "9862b079bcf496aa8d9ce9e716e65b03a3d9a70a5c2077a223630cf9dcc6d5cc",
        "jpg",
    ),
    (
        "song_sparrow-36",
        "song_sparrow",
        691561091,
        378140427,
        "vaughnshirey",
        87827,
        "a2c197f91c0543ec7bd3a9aac4867a34417b2eb2444bf3062060f9f1e3580045",
        "jpg",
    ),
    (
        "song_sparrow-37",
        "song_sparrow",
        582182748,
        321880490,
        "dougbrown",
        44795,
        "465ce1214c40efab6afdddfbfd34fc5076e5640d130b0442e8e8ac2731ec8a1f",
        "jpg",
    ),
    (
        "song_sparrow-38",
        "song_sparrow",
        704199014,
        384692464,
        "enspring",
        85269,
        "1aa25fd1ed376b9245127432925832cf9253e5d156f4f7d35298eff9a8002043",
        "jpg",
    ),
    (
        "song_sparrow-39",
        "song_sparrow",
        583754170,
        322666295,
        "rociherrera",
        97743,
        "5e093fc6d59b13bbd7ee0889613cef115ef7bb14432634d0af753a8124951737",
        "jpg",
    ),
    (
        "song_sparrow-40",
        "song_sparrow",
        590270305,
        325961498,
        "damienxw",
        133375,
        "a12a152c94ef9c27e5f6dc557e74bf57f6c9ca3015a5e28ee9c30a27ed0e7e19",
        "jpg",
    ),
    (
        "song_sparrow-41",
        "song_sparrow",
        423534602,
        237940479,
        "don54",
        25229,
        "69b0f0dfb05e8219700e3878c9b7e4a2d43d0a2df3905414630913397a565dea",
        "jpeg",
    ),
    (
        "song_sparrow-42",
        "song_sparrow",
        418537114,
        235351206,
        "memoosborne",
        95540,
        "3432cf5a95939b898989c5a962c66b2d71b92b17195d5e695020a776361af3fa",
        "jpeg",
    ),
    (
        "song_sparrow-43",
        "song_sparrow",
        280724703,
        162358184,
        "shirleymorrison",
        43015,
        "f64af9b3fc4559f251ba0f1e95bf8889bfa629123586532adeb2b81e4c2871b2",
        "jpg",
    ),
    (
        "song_sparrow-44",
        "song_sparrow",
        641009256,
        351278414,
        "robinlanark",
        49640,
        "e4af33fd4a504a876b6e7a111a5024cf4197fc888c90f2039df63e9be13d5f06",
        "jpg",
    ),
    (
        "song_sparrow-45",
        "song_sparrow",
        531314531,
        295151228,
        "steveplumb",
        128710,
        "8806861909c0833b29f43b10e9428e104deaccb99d41b42039b76f7da0ae4600",
        "jpg",
    ),
    (
        "song_sparrow-46",
        "song_sparrow",
        114487462,
        70425556,
        "leahmfulton",
        50091,
        "d0b3c9408b72514f8485b19f61116b94be002be99368db96e126d625a17bd343",
        "jpg",
    ),
    (
        "song_sparrow-47",
        "song_sparrow",
        118804811,
        72823547,
        "heibudas",
        191643,
        "8dbe5a56d5fea83b6c25257aa0f9e552ea971f1e0311f583691d24fc9668556f",
        "jpeg",
    ),
    (
        "song_sparrow-48",
        "song_sparrow",
        118498513,
        72671334,
        "seanwashington1",
        80800,
        "940762ff238d94815922d06051ca00fe4a025279b1da08921a44733084adcdb6",
        "jpeg",
    ),
    (
        "song_sparrow-49",
        "song_sparrow",
        119696805,
        73327300,
        "rambryum",
        68983,
        "65f7007d6b90654b77b7f4ec0557ff2d0f310c572368cb3492dd69f9d082e90d",
        "jpeg",
    ),
    (
        "song_sparrow-50",
        "song_sparrow",
        4251416,
        3670620,
        "swells",
        49271,
        "8c53fa5c02e603227117293720d269c59a13555abd537a83640210805fbd844d",
        "JPG",
    ),
    (
        "song_sparrow-51",
        "song_sparrow",
        303954368,
        174929167,
        "dianeclark6280",
        157478,
        "14446920b7c55472de3b172f417fbbeac312090add7356a3c947cfb16911dc95",
        "png",
    ),
    (
        "song_sparrow-52",
        "song_sparrow",
        392544542,
        221653008,
        "emckenziewhat",
        138683,
        "7fd2f3ad3c835e541a7c720372d662a8f953f88f4d8223864a03e4b0f70a42d3",
        "jpeg",
    ),
    (
        "song_sparrow-53",
        "song_sparrow",
        270614122,
        156495271,
        "radkins21",
        92783,
        "7bf8ceb5944de93cbf01aebe4d464b9acb848569c07fbe6e732b701d309f0b25",
        "jpg",
    ),
    (
        "song_sparrow-54",
        "song_sparrow",
        139634990,
        84920252,
        "raffib128",
        56293,
        "865cf92a61d89cb3420c9d3cbf585a0724df2e314918b05d705804aee356f1a5",
        "jpeg",
    ),
    (
        "song_sparrow-55",
        "song_sparrow",
        25251350,
        16733911,
        "kemper",
        54034,
        "99f19fa1a37d23712538e43f6a82339f1fd37defe283de3baacc17719f05857e",
        "jpeg",
    ),
    (
        "song_sparrow-56",
        "song_sparrow",
        139129879,
        84639025,
        "wewantashrubbery",
        104736,
        "aabbffe279d90ffdf737175cc720bea92d634483b830c90acecf920901419643",
        "jpg",
    ),
    (
        "song_sparrow-57",
        "song_sparrow",
        128242284,
        78367372,
        "stevestevens",
        112226,
        "918dfc8094d0a697beb1dcd04499cfd7b0c4b12f9dac4d915d1566ab5690b29b",
        "jpeg",
    ),
    (
        "song_sparrow-58",
        "song_sparrow",
        478733235,
        266446909,
        "saintaardvark",
        298253,
        "d3c8d3a58b44d28c6af4748e4e4195a9a2af7dc6fe91541429dc44aa3a659d21",
        "jpg",
    ),
    (
        "song_sparrow-59",
        "song_sparrow",
        623006598,
        342147975,
        "ryman56",
        60213,
        "73e18df350a69ce00158f03c7ec4fda8672539e0a7292b75f1dfa22dbec7ee30",
        "jpg",
    ),
    (
        "chipping_sparrow-00",
        "chipping_sparrow",
        198992636,
        117809422,
        "k-simpkins",
        62563,
        "cf9f3b0c1863808e21af596b2e609b047ddbc28cb2ed076625e2546425ff0adf",
        "jpg",
    ),
    (
        "chipping_sparrow-01",
        "chipping_sparrow",
        248210057,
        144599194,
        "w_mark_c",
        193389,
        "497d0a0fef81c326bcc87b5d1eb97fe987d559b8f2b71bb60fe422ae34dce150",
        "jpg",
    ),
    (
        "chipping_sparrow-02",
        "chipping_sparrow",
        156350853,
        94266719,
        "ellyne",
        142332,
        "6a60ebac34476a372b4790a87d823432cdda8f72590930b5d18ae166cf4c7ba2",
        "jpeg",
    ),
    (
        "chipping_sparrow-03",
        "chipping_sparrow",
        16128796,
        11327134,
        "reuvenm",
        70532,
        "5ad36c9cdd6c92e225a1b8ab3c04d2f65bc4e971d0243958f8ac090b0996115c",
        "jpeg",
    ),
    (
        "chipping_sparrow-04",
        "chipping_sparrow",
        391300648,
        220982684,
        "carterdorscht",
        208567,
        "c974a676c3c227c2844ff822d429c784bc87bb41f40d5074a0ebadd4c6785b21",
        "jpeg",
    ),
    (
        "chipping_sparrow-05",
        "chipping_sparrow",
        339456083,
        193252042,
        "rawcomposition",
        46329,
        "41c9260ca9107430e3a8090ba01cebf3e3c25f2dad15bea4f77c8e824d9fc496",
        "jpg",
    ),
    (
        "chipping_sparrow-06",
        "chipping_sparrow",
        40457652,
        26076708,
        "andywilson",
        156894,
        "b70edd35e00fe672a39d5f0441ea4e9bb9fac19b0f2415ae13e22160bc6091a6",
        "jpeg",
    ),
    (
        "chipping_sparrow-07",
        "chipping_sparrow",
        84151914,
        52921135,
        "davidfbird",
        145714,
        "ea65ce5ed881ded9a8157b5756fa907948f41e0eefb6735b48a1c43993e977f9",
        "jpeg",
    ),
    (
        "chipping_sparrow-08",
        "chipping_sparrow",
        523674610,
        291074747,
        "rwp84",
        47983,
        "41eac0a5fb578b089f7524c4d2e6cb38c5508aa81815d6cfc46948c081daa800",
        "jpg",
    ),
    (
        "chipping_sparrow-09",
        "chipping_sparrow",
        292695018,
        168861389,
        "tim_kirsten",
        64812,
        "cbb2a03f5dbd2209d56f1cca8b49f342dfbaf44e51fe1014ead75b2018c6678d",
        "jpeg",
    ),
    (
        "chipping_sparrow-10",
        "chipping_sparrow",
        92501489,
        57964052,
        "tniernberger",
        78380,
        "44ddb9923026a97ee77ed362bc944c3d2cbbe4fbd4636000880e81613634e6c5",
        "jpg",
    ),
    (
        "chipping_sparrow-11",
        "chipping_sparrow",
        220257388,
        129611350,
        "gcart043",
        129377,
        "1bdce32e7ac58321dd6beacc084afaf45973469215d66b61a0f7c612da3db361",
        "jpeg",
    ),
    (
        "chipping_sparrow-12",
        "chipping_sparrow",
        478480946,
        266314208,
        "russnamitz",
        61359,
        "8200cc2ec779d47be1b5afa261d934910a251a39f5620d3dc705106fb2bf5e0b",
        "jpeg",
    ),
    (
        "chipping_sparrow-13",
        "chipping_sparrow",
        538480223,
        298914072,
        "hiltonward",
        201271,
        "2a6479556f14a20a8c9a69ff0c1026fe4deb376944234d1d4429c558042c4346",
        "jpg",
    ),
    (
        "chipping_sparrow-14",
        "chipping_sparrow",
        58192408,
        36778771,
        "bradenjudson",
        22352,
        "7c630d7a5b24d94e677e563a8ecb36f57fb29babe6ac1568f11537f10a5edf75",
        "jpeg",
    ),
    (
        "chipping_sparrow-15",
        "chipping_sparrow",
        80410971,
        50636049,
        "radrat",
        55646,
        "0c71ca735502e8c94db81302ecd006428206eb16d75472eb0c706e62e2656667",
        "jpg",
    ),
    (
        "chipping_sparrow-16",
        "chipping_sparrow",
        300376697,
        172991803,
        "matthias55",
        81986,
        "7a9a5cbfb6a7d0581c98c75b5fdeb53a049d60f352eda43a39b1f9c2a3347f3b",
        "jpeg",
    ),
    (
        "chipping_sparrow-17",
        "chipping_sparrow",
        370917657,
        209626382,
        "craigmartin",
        101736,
        "feb94d319fe5b12c01e63a80dc8e44c45e15a0bfb6534ac5d2c0641205dfe1d5",
        "jpg",
    ),
    (
        "chipping_sparrow-18",
        "chipping_sparrow",
        264119989,
        152960401,
        "laurelthrone",
        194107,
        "13e7336628f9ad784757dca82557c79eda22fdce561e0b52a10a503e72979b8e",
        "jpeg",
    ),
    (
        "chipping_sparrow-19",
        "chipping_sparrow",
        220323755,
        129631264,
        "enspring",
        85687,
        "886b7e6faa39943f0c9754e4aef637e2a2f5ad57fe8e8c7ec08360ea95af45b1",
        "jpeg",
    ),
    (
        "chipping_sparrow-20",
        "chipping_sparrow",
        210319996,
        124105091,
        "bunnymom20",
        188328,
        "2f6b6f7ac5ed9d91a361c8fed102484f4fca599ec39c9a2bfe9af1df53b28899",
        "jpeg",
    ),
    (
        "chipping_sparrow-21",
        "chipping_sparrow",
        117013164,
        71832078,
        "terrimewbornagain",
        49527,
        "4456adb52cdafdefdaeb6dd8cec5955f93c1b8f40abdbb1182b2f605a0e1d486",
        "jpeg",
    ),
    (
        "chipping_sparrow-22",
        "chipping_sparrow",
        660951395,
        362143224,
        "mike_cove",
        148079,
        "41d85eaa47b3a41a97ada54728a84c0bfbfed3bcff92de3b6ba88d4c9aface48",
        "jpg",
    ),
    (
        "chipping_sparrow-23",
        "chipping_sparrow",
        691160217,
        377927837,
        "ben142",
        235922,
        "f8137e51e90f0a1964c64fac1a775beaa9ae96c90a1d9aad4f1c870f143e34fd",
        "jpg",
    ),
    (
        "chipping_sparrow-24",
        "chipping_sparrow",
        707633397,
        386482138,
        "leannestacy",
        238800,
        "3fef93babe26eed119ab3b2363902007c46a58a02f91beb0d3bc650821f0dc28",
        "jpg",
    ),
    (
        "chipping_sparrow-25",
        "chipping_sparrow",
        696064427,
        380468188,
        "rdnwoods",
        309547,
        "b56f8f043112021367cbb02861021ac2ef6d2831359315f49d1445bf92477a07",
        "jpg",
    ),
    (
        "chipping_sparrow-26",
        "chipping_sparrow",
        702557361,
        383835541,
        "seanwashington1",
        159095,
        "2940c235b38f84101504b6f50e59c359d0f13e3dc384cadbd43976d1e5e9de91",
        "jpg",
    ),
    (
        "chipping_sparrow-27",
        "chipping_sparrow",
        660337945,
        361814902,
        "kristinpiston",
        158651,
        "a2272e4168edf260001ace410774fbe874ea6b1f6632b8502efc1bc02ae0f866",
        "jpg",
    ),
    (
        "chipping_sparrow-28",
        "chipping_sparrow",
        546647394,
        303298450,
        "kzoebel",
        86457,
        "c37a4aff4263de570ca82f94de92f82d9f9711922e5880547af836104f1883df",
        "jpg",
    ),
    (
        "chipping_sparrow-29",
        "chipping_sparrow",
        713495256,
        389515472,
        "lina47741",
        287986,
        "52e8a37dca8d985a57fdd0d90205e1d813388d961dbf4280d5cbccca9c6af5d3",
        "jpg",
    ),
    (
        "chipping_sparrow-30",
        "chipping_sparrow",
        269951645,
        156136769,
        "conhawn",
        106540,
        "11b48d8bf97084d9ff57aa69517b61539f03b72525022930192330fcb05a160c",
        "jpeg",
    ),
    (
        "chipping_sparrow-31",
        "chipping_sparrow",
        535714979,
        297462154,
        "dissectedfrog",
        200498,
        "cd2f2e1be8c76240a1eeff9535e9a60b8f4fb9d771c7e487d65cde48b7e24cf6",
        "jpg",
    ),
    (
        "chipping_sparrow-32",
        "chipping_sparrow",
        557301829,
        308874987,
        "mar_y_sierra_silvestre",
        115174,
        "ce58e87584df623a86643705b08750228903705ca9b02d93b0206401a24eee9e",
        "jpg",
    ),
    (
        "chipping_sparrow-33",
        "chipping_sparrow",
        652401709,
        357680572,
        "dianeclark6280",
        46782,
        "b77c0eea6899d49f8ab97fd1ad6fc772b6055e5195628fa9f6f989e2d51cedd1",
        "jpg",
    ),
    (
        "chipping_sparrow-34",
        "chipping_sparrow",
        657945260,
        360569392,
        "jasonleduc",
        36363,
        "e3471104742fa68b9eeb149255b0136c53306df23d14f2b2633cea80c9e46eea",
        "jpg",
    ),
    (
        "chipping_sparrow-35",
        "chipping_sparrow",
        434553038,
        243681621,
        "stariplativky",
        57489,
        "e253bd9d70dea9d0438b795f6bebcbc2b26c48b16a90729f0d3155695fa812fd",
        "jpg",
    ),
    (
        "chipping_sparrow-36",
        "chipping_sparrow",
        207498409,
        122551474,
        "askalotl",
        98446,
        "d84e28017be8142c22758e59f14b8c52990e79584a68e6f929123aab4997ac68",
        "jpeg",
    ),
    (
        "chipping_sparrow-37",
        "chipping_sparrow",
        179570776,
        106840941,
        "artemis224",
        138087,
        "74cad843089d2aada7a124f9e704ae1c1419b7cfcade7e2ae29e2c4eda0a6d36",
        "jpg",
    ),
    (
        "chipping_sparrow-38",
        "chipping_sparrow",
        81287617,
        51172999,
        "thyg",
        55659,
        "e1f075addb4bd1feea3ab256e37945a464d2e8c77768b7be131d0655382268d7",
        "jpg",
    ),
    (
        "chipping_sparrow-39",
        "chipping_sparrow",
        343116928,
        195090930,
        "umamimomma",
        52983,
        "a8948a0189d19b3d7b8df65271f4844b14bd5118f510d0b9ef458c942ac41ce2",
        "jpg",
    ),
    (
        "chipping_sparrow-40",
        "chipping_sparrow",
        480169680,
        267207764,
        "cvharris",
        144339,
        "d1ab8643c48f2ea823f9b61843016795def21b290c35ce3d6045e4933b5dfa0a",
        "jpeg",
    ),
    (
        "chipping_sparrow-41",
        "chipping_sparrow",
        352953833,
        200088501,
        "aster-asti",
        105786,
        "82c35e04e4e22fe35c9b364bce7737fa5226cb5c469c761adfc61ef0956da9db",
        "jpg",
    ),
    (
        "chipping_sparrow-42",
        "chipping_sparrow",
        622149696,
        341721858,
        "wilderbombyx",
        293652,
        "227c39a25941a81e9551b5ccef39f95d320e7daa02c96de247123ba62f7b9824",
        "jpg",
    ),
    (
        "chipping_sparrow-43",
        "chipping_sparrow",
        503555630,
        280377982,
        "jtdavis05",
        42352,
        "24ccb474659505cd785502295f385ec369a36c6248296aff85261ed3475a4372",
        "jpeg",
    ),
    (
        "chipping_sparrow-44",
        "chipping_sparrow",
        509135437,
        283351340,
        "martyndrabik",
        125948,
        "1604bf4f5512207c59637a84c78842285c40f76a4cf5335bf6d39afe896ed03c",
        "jpg",
    ),
    (
        "chipping_sparrow-45",
        "chipping_sparrow",
        8076820,
        6402608,
        "msieges",
        166588,
        "6c379f94141a0bcdeb8c9743ea140ded50f8c7864baecd6cef0ee1c724b9fd80",
        "jpeg",
    ),
    (
        "chipping_sparrow-46",
        "chipping_sparrow",
        131917390,
        80486493,
        "tom_lazar",
        137355,
        "10ab0a9c5af2ca2203a18e4e3df799dcf129494d7b637a3c46ffdcf0f3b3856c",
        "jpeg",
    ),
    (
        "chipping_sparrow-47",
        "chipping_sparrow",
        141348323,
        85865218,
        "j-dehoog",
        71473,
        "d3fae9ca7f515c43c216d749f085231d19ddfd101bd37cc4ffae25fc946a34fb",
        "jpeg",
    ),
    (
        "chipping_sparrow-48",
        "chipping_sparrow",
        28879424,
        18850970,
        "andy71",
        194983,
        "ea5af97ddef4f743596c176653c86e457f6bddde65d86f5ab16ddff939574683",
        "jpeg",
    ),
    (
        "chipping_sparrow-49",
        "chipping_sparrow",
        148831027,
        90084486,
        "ian-wolfe",
        181942,
        "cfab51f0c0598120f5312b354ae249bafb15124dd74723a8c29cdd7319206e49",
        "jpg",
    ),
    (
        "chipping_sparrow-50",
        "chipping_sparrow",
        641370647,
        351455126,
        "hrachski",
        130262,
        "9bc25019bf6a71d73daac63b810d18039d360887504014525598cffce2035908",
        "jpg",
    ),
    (
        "chipping_sparrow-51",
        "chipping_sparrow",
        648268506,
        355462611,
        "potatocthulhu",
        193138,
        "f526135f7b71e0f9f8d80c17d8e6c1a3cd89ba068333b9104e66d02ced7a7eaa",
        "jpg",
    ),
    (
        "chipping_sparrow-52",
        "chipping_sparrow",
        645412478,
        353758550,
        "ctlqh",
        154844,
        "e34228baa058ddb18c6aacf193c20574dab67d8c1f4d6940eec6e1acb03d993b",
        "jpg",
    ),
    (
        "chipping_sparrow-53",
        "chipping_sparrow",
        121777317,
        74504842,
        "hickl",
        48576,
        "a7f0980bf777cacb9f61855d157b94d895f6dd629485b2a038ce20519c53856c",
        "jpg",
    ),
    (
        "chipping_sparrow-54",
        "chipping_sparrow",
        147992761,
        89606238,
        "benkeen",
        461732,
        "8b6e8f03650c270624ebf82f85dd0b37db3e693157d4d3d7a1787c855846da9d",
        "png",
    ),
    (
        "chipping_sparrow-55",
        "chipping_sparrow",
        27976676,
        18310312,
        "schoenitz",
        66424,
        "64a0186d15abaf1537a34bd329b67625f7580b7aa544dded6bc626dd389b72fe",
        "jpg",
    ),
    (
        "chipping_sparrow-56",
        "chipping_sparrow",
        190706448,
        112884695,
        "jbeusmans",
        109296,
        "1777945eeb6f916ec6f9c0dc10c81014e8dc0ba22d87268e7c7dc1173193b1e6",
        "jpeg",
    ),
    (
        "chipping_sparrow-57",
        "chipping_sparrow",
        184174664,
        109291439,
        "chrismcv",
        91566,
        "c5d9a1dbaf1fad839680c417c7be7d4a680396e38e7c2aba0ee78fd213f7d32c",
        "jpeg",
    ),
    (
        "chipping_sparrow-58",
        "chipping_sparrow",
        72765715,
        45879059,
        "henryfrye",
        146855,
        "f0b957698f5a600e3a6c45471848013f9c74d1602895e40d513e2116a9b7320a",
        "jpg",
    ),
    (
        "chipping_sparrow-59",
        "chipping_sparrow",
        74100878,
        46721618,
        "aredmer",
        132045,
        "ce6f4d4324d684cfc0f40d6fe9c4a2b03a7024080e7448083ae7e3fc0c06ac6e",
        "jpg",
    ),
    (
        "white_throated_sparrow-00",
        "white_throated_sparrow",
        339621218,
        193338380,
        "rawcomposition",
        31152,
        "d1c08bfaca721bf0873437455b4cc010c6860d08b4777136c775007b0b9d07b6",
        "jpg",
    ),
    (
        "white_throated_sparrow-01",
        "white_throated_sparrow",
        166821399,
        99992799,
        "dziakj1",
        125954,
        "c595a41fbc8948b0d918b59117340dc320e2ba80d29e92bb9dacaaed5b404852",
        "jpeg",
    ),
    (
        "white_throated_sparrow-02",
        "white_throated_sparrow",
        469820434,
        261505977,
        "joy4birds",
        111767,
        "7ce091492c73c68395667bb45578dfff11457511b0958d1d0d320d1df3e55ceb",
        "jpg",
    ),
    (
        "white_throated_sparrow-03",
        "white_throated_sparrow",
        260413022,
        150969515,
        "andywilson",
        45958,
        "fb7c533c92239775da6a5b353ff398f6bdd8f65c13445fc1dd72986af5a46466",
        "jpeg",
    ),
    (
        "white_throated_sparrow-04",
        "white_throated_sparrow",
        99351488,
        62040646,
        "bradenjudson",
        23399,
        "e103968e2a6c9efb6f0bcb548a6457aef950a4a720838a624b6796b90411538e",
        "jpeg",
    ),
    (
        "white_throated_sparrow-05",
        "white_throated_sparrow",
        628148203,
        344731686,
        "lavenderdame",
        106872,
        "36379abf3af51d865ba6e6804ba3dd48fc9efd12ecc0b7d97e03fc04a16178a8",
        "jpg",
    ),
    (
        "white_throated_sparrow-06",
        "white_throated_sparrow",
        104660609,
        65043951,
        "allan7",
        42443,
        "10791891945e83a0c908c14fea07a257a0f25f51c0e708bee35184213833a635",
        "jpeg",
    ),
    (
        "white_throated_sparrow-07",
        "white_throated_sparrow",
        250718938,
        145903421,
        "stevestevens",
        138735,
        "bd52e6d2f48c247f72db0fa393ec4fa13af8510c95f0ca9134cbef71caddb6d4",
        "jpeg",
    ),
    (
        "white_throated_sparrow-08",
        "white_throated_sparrow",
        15105971,
        10793852,
        "schylerbrown",
        31467,
        "6496e7e6d3abf13b1a538f769e3cc402280cf1e9a2a1ebdf81c68dcfd7ed01e2",
        "jpeg",
    ),
    (
        "white_throated_sparrow-09",
        "white_throated_sparrow",
        171460784,
        102554447,
        "w_mark_c",
        251287,
        "3828c41af9209b408fe0d8ec6541edf35d13fc730b74fd27a82980d4f3771137",
        "jpg",
    ),
    (
        "white_throated_sparrow-10",
        "white_throated_sparrow",
        244260317,
        142471526,
        "deejay",
        75694,
        "d67efb1ec61f6700b8a6c6552e2da9cd981e68ae81af16d6f1e0ef17fcaff84f",
        "jpeg",
    ),
    (
        "white_throated_sparrow-11",
        "white_throated_sparrow",
        194732675,
        115373159,
        "wildreturn",
        128989,
        "7ed4bde480b35734576bb4c5f9d1453e77c1f095380432b43b1b682050255831",
        "jpeg",
    ),
    (
        "white_throated_sparrow-12",
        "white_throated_sparrow",
        341990462,
        194541980,
        "ethologist",
        149260,
        "3e2710fac082cc347e6cd114d42d39f25ec47b82c25936922232eba916808cdb",
        "jpg",
    ),
    (
        "white_throated_sparrow-13",
        "white_throated_sparrow",
        267625208,
        154862375,
        "laurelthrone",
        148616,
        "0cac1bc7053c891ce5ae1c33e3f94b69f1b19b958bd08c686db2ccaa7ba0fbd1",
        "jpeg",
    ),
    (
        "white_throated_sparrow-14",
        "white_throated_sparrow",
        330539586,
        188793537,
        "efalquet",
        119214,
        "c3db4583124472b28dbfb4c6fd9b3829e451d508a6b0a095eccc28f17e7982b4",
        "jpeg",
    ),
    (
        "white_throated_sparrow-15",
        "white_throated_sparrow",
        193091403,
        114346779,
        "ian-wolfe",
        73751,
        "e28974c8782fccb00f5ea5420140e01248ead859f151563a95a26bf67acba479",
        "jpg",
    ),
    (
        "white_throated_sparrow-16",
        "white_throated_sparrow",
        691050773,
        377873006,
        "dinomariobob",
        167744,
        "cc5e25afa5041ce2d6ea4e0d726793843f3a867f30b8d9ccf55892d6617da887",
        "jpg",
    ),
    (
        "white_throated_sparrow-17",
        "white_throated_sparrow",
        440986574,
        246671481,
        "suzannehale",
        141857,
        "4a354c189225de2e7b4e94a6df9cbd3a671dac0c8a7d2d4c75e933f93d8b83bf",
        "jpeg",
    ),
    (
        "white_throated_sparrow-18",
        "white_throated_sparrow",
        113217820,
        69733707,
        "kemper",
        97802,
        "4de7da3ac53a086f2c343555be5a2b62a683715db9f80636a76673cc380fa7f6",
        "jpeg",
    ),
    (
        "white_throated_sparrow-19",
        "white_throated_sparrow",
        575307217,
        318327478,
        "k-simpkins",
        90416,
        "af2b4e72a098bd90b920c8a49632e1bbff18b73954d8ac541ca81ed4baf0530b",
        "jpg",
    ),
    (
        "white_throated_sparrow-20",
        "white_throated_sparrow",
        340420076,
        193765555,
        "don54",
        62714,
        "7f6eea434fc700166af1d343951d15f3a9c83eff06cfb0518c3c4291019a5556",
        "jpeg",
    ),
    (
        "white_throated_sparrow-21",
        "white_throated_sparrow",
        562344160,
        311510652,
        "rrfc",
        45842,
        "f7c6394649765db6e3313a03ae013291db1ae20d89d3b8f0bdf6641757f31ed0",
        "jpg",
    ),
    (
        "white_throated_sparrow-22",
        "white_throated_sparrow",
        74598266,
        47039878,
        "ianrwhyte",
        141502,
        "b0633999572a4499a80310955ab928a86f4fe08774e669b4c4949cd26254428b",
        "jpeg",
    ),
    (
        "white_throated_sparrow-23",
        "white_throated_sparrow",
        588001234,
        324825124,
        "portablecity",
        370187,
        "11e3a97f43de5d61266d15028fe9023eef3900cfea2d027bd94ad847ecba9607",
        "jpg",
    ),
    (
        "white_throated_sparrow-24",
        "white_throated_sparrow",
        694103481,
        379467387,
        "memoosborne",
        120656,
        "82684f82f62f6e036e206eec315fff8e1ba36d308b667eba06750b75d1fada3a",
        "jpg",
    ),
    (
        "white_throated_sparrow-25",
        "white_throated_sparrow",
        655749108,
        359408087,
        "russnamitz",
        58449,
        "2a7d2b4226f1941d0950adc04bfd8fe62345f1138c521d0f5f0adf8bc5838da1",
        "jpg",
    ),
    (
        "white_throated_sparrow-26",
        "white_throated_sparrow",
        599110899,
        330395670,
        "jd_flores",
        121343,
        "5eeef4f5533a4b0a0222dcf7be000a2b34f3331f3be2b939703155ba8c8ba3ae",
        "jpg",
    ),
    (
        "white_throated_sparrow-27",
        "white_throated_sparrow",
        653888173,
        358443775,
        "toknowtheland",
        144197,
        "20a4933b00967fe4488296b2b6e89c12ecbcca0e40da591bd95927cca46ae2f7",
        "jpg",
    ),
    (
        "white_throated_sparrow-28",
        "white_throated_sparrow",
        295037357,
        170132018,
        "sturuss",
        117077,
        "d309f75ad5f4008c906ba7b3d6138aa017f919a0a17a1d5b431fa7ab47d5f2d3",
        "jpg",
    ),
    (
        "white_throated_sparrow-29",
        "white_throated_sparrow",
        405042885,
        228270930,
        "carterdorscht",
        122157,
        "f1fb32bcfb78f66f50ccd20f2418832afe72c7a5847bdeb8e4dc19546455cfaf",
        "jpeg",
    ),
    (
        "white_throated_sparrow-30",
        "white_throated_sparrow",
        84971412,
        53425343,
        "davidfbird",
        168442,
        "95c5363704d17d6076f3c7a4c9f5e41d235edf7d5dc9d6022116774f2fd544e6",
        "jpg",
    ),
    (
        "white_throated_sparrow-31",
        "white_throated_sparrow",
        476335868,
        265210970,
        "aster-asti",
        91852,
        "cd300cbc823e8ebc0902b7bd80c99eb323e5900bbabf63303b0d9399624c3e54",
        "jpg",
    ),
    (
        "white_throated_sparrow-32",
        "white_throated_sparrow",
        481170792,
        267732071,
        "kcthetc1",
        46021,
        "2194a78339aae99549df56d2d3d7fc91e0bf98a371e7d0da7958b4312fa3b487",
        "jpeg",
    ),
    (
        "white_throated_sparrow-33",
        "white_throated_sparrow",
        623602892,
        342440773,
        "imperialwoodpecker26",
        112678,
        "66653f47ce6a54d38cc1f54b0a85e965f5c5ee7cce65b02c78c1c185cb586c85",
        "jpg",
    ),
    (
        "white_throated_sparrow-34",
        "white_throated_sparrow",
        622732944,
        342010560,
        "tom_lazar",
        34257,
        "584070c4b5231db49087d1507aabec8f51adc4e09dcdda8b93b3a78159d47146",
        "jpg",
    ),
    (
        "white_throated_sparrow-35",
        "white_throated_sparrow",
        505347090,
        281327144,
        "erikschiff",
        85652,
        "91d50bda6afc3cd5596bf4d3176999bf473c31fb6680b64445ec1cbeaabea847",
        "jpg",
    ),
    (
        "white_throated_sparrow-36",
        "white_throated_sparrow",
        7072170,
        5706496,
        "jmutter",
        171582,
        "de90ea2739e8c65ca89f0e511abaf44219501e18377eb49d9e5c7f10d3a6d5b2",
        "jpeg",
    ),
    (
        "white_throated_sparrow-37",
        "white_throated_sparrow",
        16303076,
        11431253,
        "robw",
        61348,
        "e7d81cfe027c7b0914c79a0431f0ce1431db1ebe78368a2fe5f31e2aab59c3c7",
        "jpg",
    ),
    (
        "white_throated_sparrow-38",
        "white_throated_sparrow",
        7149657,
        5761350,
        "bradleysaul",
        108304,
        "adf755b8393fb6b4db99e788d4112c2c72197c33ecffeec2ca7b3ba5dd9d650e",
        "jpg",
    ),
    (
        "white_throated_sparrow-39",
        "white_throated_sparrow",
        27513220,
        17998912,
        "mefisher",
        49587,
        "f1716d49782f70455ec97ecdda634505b6b2afc375105b1d779353e704dda07b",
        "jpg",
    ),
    (
        "white_throated_sparrow-40",
        "white_throated_sparrow",
        642121459,
        351843522,
        "jeffcherry",
        53348,
        "c9e82d7e47ca2e51c856f9bd4d8f2965d8f69948f081c08c9fa6ebf44fcde711",
        "jpg",
    ),
    (
        "white_throated_sparrow-41",
        "white_throated_sparrow",
        642355950,
        351962427,
        "robinlanark",
        54504,
        "c0da16d5146afac102fa274fffea2e0b203282078ed692f3b7ad4a1340b0d4cd",
        "jpg",
    ),
    (
        "white_throated_sparrow-42",
        "white_throated_sparrow",
        638199267,
        349826931,
        "wildaboutwildlife",
        32981,
        "845f7d38d6f35d2e6e9dca61234c9566778fec1e274500c97e47977fcecd232f",
        "jpg",
    ),
    (
        "white_throated_sparrow-43",
        "white_throated_sparrow",
        119839965,
        73407944,
        "terrimewbornagain",
        61951,
        "dbfbaaa2f72caa812c026122f41512caef00f8f8fdd21c2307fcdf9d79921757",
        "jpeg",
    ),
    (
        "white_throated_sparrow-44",
        "white_throated_sparrow",
        6236370,
        5078910,
        "ruggedbynature",
        126403,
        "3d6030593bf57bc5734fd6c0706c1e350bb117a32370a16b194ef4488668f123",
        "jpeg",
    ),
    (
        "white_throated_sparrow-45",
        "white_throated_sparrow",
        32221828,
        20882923,
        "christinan",
        132485,
        "b92ce41128bb1c5b53a0c487b46ba80eea83b9a443ee9c72f100005e5d80e411",
        "jpeg",
    ),
    (
        "white_throated_sparrow-46",
        "white_throated_sparrow",
        36598789,
        23643589,
        "jessm-c",
        45861,
        "eda75171590418596e372c6cba058364acb64f1279d250dff6c5df61fa17868b",
        "jpeg",
    ),
    (
        "white_throated_sparrow-47",
        "white_throated_sparrow",
        163198257,
        98065830,
        "ellyne",
        109261,
        "864fb42a82561684c1ac22ae71bd023ddeb15c6052b7732ebd2570a98b552a1a",
        "jpeg",
    ),
    (
        "white_throated_sparrow-48",
        "white_throated_sparrow",
        162150240,
        97493574,
        "c_burns802",
        188464,
        "0fb2fb2e04bca82c436315b77a974adc665d2d1fb4e419dd61afa068ab412254",
        "jpeg",
    ),
    (
        "white_throated_sparrow-49",
        "white_throated_sparrow",
        161756778,
        97282492,
        "raffib128",
        67237,
        "77384a65eb120bdb345c3015cb8c22fcec9c89df49b0e0c00954e3476834da9f",
        "jpeg",
    ),
    (
        "white_throated_sparrow-50",
        "white_throated_sparrow",
        165146429,
        99107203,
        "philippthompson",
        84518,
        "8582544ca8b3cb1c73174fc42a9d66971fdae0c90cd4739083eb4ef4983eb6e7",
        "jpeg",
    ),
    (
        "white_throated_sparrow-51",
        "white_throated_sparrow",
        177451999,
        105722646,
        "natepow",
        44983,
        "c81a68c46264d1a5fcae3042e7ab24653c3615665cef6293c31622d5f342d840",
        "jpg",
    ),
    (
        "white_throated_sparrow-52",
        "white_throated_sparrow",
        67630852,
        42591255,
        "schoenitz",
        138616,
        "34594c49a4f601b17e49dfc4d4098db7b73ff5a69d3830b773378e4d695ff181",
        "jpg",
    ),
    (
        "white_throated_sparrow-53",
        "white_throated_sparrow",
        99106006,
        61898962,
        "glennberry",
        122576,
        "fd8cd1c0b0c1fdd7636b8588c86c6d4f3cddc233e9bd4bccade5e2ed275d0e28",
        "jpg",
    ),
    (
        "white_throated_sparrow-54",
        "white_throated_sparrow",
        102162606,
        63634844,
        "eug302",
        73329,
        "692114df14fb8a75fdf38428a196467a4a394e5ded95605fbb48cf64903ccb1b",
        "jpeg",
    ),
    (
        "white_throated_sparrow-55",
        "white_throated_sparrow",
        370403033,
        209307218,
        "kuykenwil",
        124312,
        "d0700068b06ad0f79eeac331a92e5d633bd76be264c3d8801a9a6eb301193969",
        "jpg",
    ),
    (
        "white_throated_sparrow-56",
        "white_throated_sparrow",
        500790892,
        278893388,
        "quillipede",
        186273,
        "0db918123c7f529ca76c04129886eb9aeb947bd3e90cdfd968c4605931c16f85",
        "jpeg",
    ),
    (
        "white_throated_sparrow-57",
        "white_throated_sparrow",
        364738073,
        205369541,
        "sandra1142",
        144424,
        "4cd383751871426904f4d8ff7526a312ba33a6c894faf7121c6ec0ff6ce0d7b1",
        "jpeg",
    ),
    (
        "white_throated_sparrow-58",
        "white_throated_sparrow",
        374502828,
        211884837,
        "halliefromcali",
        62476,
        "0073652fddd7378dc936f280c18d7e44c4a7160aadc2f051e7b5e2896b1a476c",
        "jpg",
    ),
    (
        "white_throated_sparrow-59",
        "white_throated_sparrow",
        109241547,
        67591707,
        "blkillin",
        130154,
        "e5b8c1caa0f7cc7926d46eac42df582ccb0882b4a0733fc0740b8b4597a4e014",
        "jpeg",
    ),
    (
        "dark_eyed_junco-00",
        "dark_eyed_junco",
        172110799,
        102901486,
        "schylerbrown",
        182973,
        "185209c7a1111fc626a068e136ec3cfcdff3174d15f1c60af209d7b32fe7bef9",
        "jpeg",
    ),
    (
        "dark_eyed_junco-01",
        "dark_eyed_junco",
        46691943,
        29901256,
        "haida_gwaii",
        46823,
        "a92dca21e6e58c375fc313f0d1da2c86c11408beb0df8fd96fc60ae035ae80d5",
        "jpg",
    ),
    (
        "dark_eyed_junco-02",
        "dark_eyed_junco",
        707222551,
        386266764,
        "ben142",
        289608,
        "7bf320edf4d34a4848f3e9d175cfafa66cc5bab6a1a8f97c41c670263de3ff89",
        "jpg",
    ),
    (
        "dark_eyed_junco-03",
        "dark_eyed_junco",
        346777340,
        196961623,
        "zacharyfoster",
        46712,
        "ea8f9f0eebb4d2f86193c705344a8fab09bf634bc2217a0874ee57c4f0f5b4ab",
        "jpg",
    ),
    (
        "dark_eyed_junco-04",
        "dark_eyed_junco",
        8793471,
        6892999,
        "truthseqr",
        45302,
        "f8241eab39797c4e097a1432b13658466287d61ed515448457907c17e60cf5e2",
        "jpeg",
    ),
    (
        "dark_eyed_junco-05",
        "dark_eyed_junco",
        192557376,
        114006980,
        "k-simpkins",
        172398,
        "206f012add8a5d0fa434e07c51f1e7bd73a60c4d299a2202163fc662e1b03479",
        "jpeg",
    ),
    (
        "dark_eyed_junco-06",
        "dark_eyed_junco",
        243303909,
        141959574,
        "andywilson",
        71321,
        "ee5308b6f93fb40a6d795a7d8ca6f2ef55b4844513f4268ef34827e1c5e4c4a6",
        "jpeg",
    ),
    (
        "dark_eyed_junco-07",
        "dark_eyed_junco",
        274980085,
        159160633,
        "andy71",
        51209,
        "c54b45ca7fdc635bdb31eb89166c9fce84f0b0f8f0c17331fd5e42b582633c9b",
        "jpeg",
    ),
    (
        "dark_eyed_junco-08",
        "dark_eyed_junco",
        213798376,
        126031618,
        "nathanael15",
        57646,
        "d6b9e7d2dcc63cdd88b47cce328149aa9e8eb486ee2a5fc0f08b90601f2c7d9b",
        "jpg",
    ),
    (
        "dark_eyed_junco-09",
        "dark_eyed_junco",
        63482066,
        39977347,
        "chrisleearm",
        41870,
        "90573e814dbde965c70a934d9702110c40670aa22cc9f80ff8849db877d1fd72",
        "jpeg",
    ),
    (
        "dark_eyed_junco-10",
        "dark_eyed_junco",
        458710472,
        255914048,
        "joy4birds",
        90973,
        "fd25ecb1bf86896b84e9e4e8329c742011011e6883631f47dbe3744ba7463153",
        "jpg",
    ),
    (
        "dark_eyed_junco-11",
        "dark_eyed_junco",
        20784016,
        14046286,
        "gambolingquail",
        93957,
        "a454d6f987153317c9c57c05019b08ab0ebebc46d3cf72356b014231b69a1e9d",
        "jpeg",
    ),
    (
        "dark_eyed_junco-12",
        "dark_eyed_junco",
        332842159,
        189933284,
        "jan-konilu",
        110145,
        "d18706f536164a72a4ec9bacf47437edd89d6b547f86eefdf10ed0166cebf0dd",
        "jpeg",
    ),
    (
        "dark_eyed_junco-13",
        "dark_eyed_junco",
        513004508,
        270404136,
        "thevertebratepokedex",
        141252,
        "854f75e7527932d0409e14756725388de9913c0b450536da717e54d683de169f",
        "jpg",
    ),
    (
        "dark_eyed_junco-14",
        "dark_eyed_junco",
        593499256,
        327583635,
        "orionid",
        108583,
        "7eab4cea6d46faa878732d82c5c2ece63ecc3e66781ea08d0bd950eee581d16d",
        "jpg",
    ),
    (
        "dark_eyed_junco-15",
        "dark_eyed_junco",
        256948665,
        149140989,
        "igor322",
        86925,
        "9339b8cf4aa347b4eb176ef1209dfaa4fcb11e277889cc17de399d6af2c8633a",
        "jpeg",
    ),
    (
        "dark_eyed_junco-16",
        "dark_eyed_junco",
        12533281,
        9255403,
        "braincellsgone",
        40857,
        "be34c51655f59612858d888b05a4f927da65e1aede850c9a5f1be68fa00bfcbe",
        "jpg",
    ),
    (
        "dark_eyed_junco-17",
        "dark_eyed_junco",
        12580989,
        9282523,
        "artemis224",
        216253,
        "15198c123c01fa0f8c03edc086408c8d95ffa553083c380e8ae236dbb7056093",
        "jpg",
    ),
    (
        "dark_eyed_junco-18",
        "dark_eyed_junco",
        63590391,
        40041059,
        "bobbyblackmore",
        82853,
        "87746be0bca30fc40ddba739c1a35ed1fdd41882ebfe3561518b69d285f5ae5d",
        "jpeg",
    ),
    (
        "dark_eyed_junco-19",
        "dark_eyed_junco",
        106718005,
        66230973,
        "vicki936",
        41475,
        "e6903e9a69e987c45edd468ace0bf71adf8252063ff5d39c2130409d8fea68be",
        "jpeg",
    ),
    (
        "dark_eyed_junco-20",
        "dark_eyed_junco",
        611049594,
        336277421,
        "skylar_schell",
        15425,
        "7e876479febb3d44b1e494a25f778efe4e8bfd323279031cb08f9f6fe95f742e",
        "jpg",
    ),
    (
        "dark_eyed_junco-21",
        "dark_eyed_junco",
        591147962,
        326403963,
        "toknowtheland",
        106373,
        "07770f3313dc969802355fa8b3e62a86edb115a64771bd38938e1fab33220c39",
        "jpg",
    ),
    (
        "dark_eyed_junco-22",
        "dark_eyed_junco",
        459696953,
        256398190,
        "w_mark_c",
        262126,
        "5025e757c026e02160ff3f0e82f1f4434b888a6c9689e7ac76274c1fcec64f0c",
        "jpg",
    ),
    (
        "dark_eyed_junco-23",
        "dark_eyed_junco",
        484171775,
        269292238,
        "aschuman",
        59459,
        "89dd933d12cf21c0e73eb01e96a9279e2e57b36676a72a9c860281d0d7703282",
        "jpg",
    ),
    (
        "dark_eyed_junco-24",
        "dark_eyed_junco",
        469746672,
        261469406,
        "shannon_j",
        74167,
        "ec4c7ce15aa1c4eadd3bc5d46d433baf9560d6a0e4593d8d57aab8fa9bec9e18",
        "jpg",
    ),
    (
        "dark_eyed_junco-25",
        "dark_eyed_junco",
        357382685,
        202362003,
        "dougbrown",
        56324,
        "70275d8e07192c99e121b67d206de6823b194f043e65b9766f1f9ce772953c78",
        "jpeg",
    ),
    (
        "dark_eyed_junco-26",
        "dark_eyed_junco",
        7660371,
        6119391,
        "jeffreyleeisanaturalist",
        108394,
        "0f667e9af8e8d3585807da58a2f315c5dbcb7eece5f6862006fb9da68a4bef42",
        "jpg",
    ),
    (
        "dark_eyed_junco-27",
        "dark_eyed_junco",
        437957476,
        245430473,
        "eug302",
        106520,
        "394abfbcf192ecf5a76cb2f19dfa2a62070fa3874b084d1468219d6ee5db6cbf",
        "jpeg",
    ),
    (
        "dark_eyed_junco-28",
        "dark_eyed_junco",
        339439778,
        193239701,
        "rawcomposition",
        32258,
        "73fad57ed975df01c529e77eaca9eab9b00741ea2cfc91358856aefca620a10a",
        "jpg",
    ),
    (
        "dark_eyed_junco-29",
        "dark_eyed_junco",
        634100291,
        347744522,
        "zorthesosen",
        169264,
        "932237001fe68e3d6d19c496fd448b8a82f254316211ac6f9d35708a1395ed81",
        "jpg",
    ),
    (
        "dark_eyed_junco-30",
        "dark_eyed_junco",
        6198359,
        5055484,
        "glmory",
        108168,
        "d31767f42eaaf7f3527133fffb9a0271a9dbbe66fc036d5c694b605563d09965",
        "jpg",
    ),
    (
        "dark_eyed_junco-31",
        "dark_eyed_junco",
        6485959,
        5243663,
        "fake_id",
        148254,
        "ce4dab6c5b37b7ebac07f69d4040d4686fea9fd4b753f91bdf15a3a440820712",
        "jpg",
    ),
    (
        "dark_eyed_junco-32",
        "dark_eyed_junco",
        246318772,
        143610209,
        "wildreturn",
        62090,
        "839e57443659659522ed65dfd054a8fc0a6bfe400518d9f9a7519383d33a2c0d",
        "jpeg",
    ),
    (
        "dark_eyed_junco-33",
        "dark_eyed_junco",
        253199424,
        147174876,
        "sean579",
        121660,
        "6efc17eb4c0b184d64a605a76d3734d9bff5d26e8c8609d2f0b0dd4ccc8b3faa",
        "jpeg",
    ),
    (
        "dark_eyed_junco-34",
        "dark_eyed_junco",
        244315013,
        142500114,
        "enspring",
        63514,
        "1d5012c2b4504b4347fef36d8764eebd1bf07649a14cf874e6d0b3eba9fb3c4f",
        "jpg",
    ),
    (
        "dark_eyed_junco-35",
        "dark_eyed_junco",
        256191686,
        148740024,
        "nartb",
        194532,
        "e88b1449a989fc62c5da1c592e4987fa4db71eadf1ef29c0c5c1159b8f77a298",
        "jpg",
    ),
    (
        "dark_eyed_junco-36",
        "dark_eyed_junco",
        241013731,
        140770209,
        "nana10",
        196479,
        "fc773ca1b06e0e3673aac5c2213c66981ede11bf3ad4a0cdef2102c4f1d99843",
        "jpeg",
    ),
    (
        "dark_eyed_junco-37",
        "dark_eyed_junco",
        240712071,
        140608440,
        "saintaardvark",
        129995,
        "94c4dc30ca31804d6645a6be802cf87ccaffe974f0d3cb554b4fb3e9696ec6a5",
        "jpg",
    ),
    (
        "dark_eyed_junco-38",
        "dark_eyed_junco",
        242909841,
        141750179,
        "calinsdad",
        61773,
        "c49c5d22969552de0fc04ff95eefb8a2ffc76df528477b93c2e2c38cdebe8f6c",
        "jpg",
    ),
    (
        "dark_eyed_junco-39",
        "dark_eyed_junco",
        109385584,
        67667584,
        "ninetoes",
        64185,
        "64e9ee29cf017c292e1711b4c6482488a74c8f652e88e8b7c1df883dc91d75e2",
        "jpg",
    ),
    (
        "dark_eyed_junco-40",
        "dark_eyed_junco",
        102027986,
        63561089,
        "michaelnaumoff",
        114346,
        "eccaa3c60fe575e28fc60f8f6896a99065bff4aee0a4fdc057d7220e7870ca4d",
        "jpg",
    ),
    (
        "dark_eyed_junco-41",
        "dark_eyed_junco",
        105652410,
        65622040,
        "don54",
        159856,
        "5c4282e5e85af7a886eceeadc90977bcd11bf26658176ba5ef8182d2f4a2bfe1",
        "jpeg",
    ),
    (
        "dark_eyed_junco-42",
        "dark_eyed_junco",
        267851501,
        154986181,
        "shanebustapbj",
        203657,
        "217002a366779db0ccc3fddffa71e79472b1fcaa5d444049971dac4c31c910f5",
        "jpeg",
    ),
    (
        "dark_eyed_junco-43",
        "dark_eyed_junco",
        268597047,
        155389350,
        "j-dehoog",
        54916,
        "c0890844944f3d603f8220b0a75213051766c6a45b2ba1fed51381134832f69f",
        "jpeg",
    ),
    (
        "dark_eyed_junco-44",
        "dark_eyed_junco",
        515881506,
        286951393,
        "jubileej",
        108221,
        "9cbecae915f1c96dcc31d5fd77cb5902181152cc3b918693191cc11fce18822c",
        "jpg",
    ),
    (
        "dark_eyed_junco-45",
        "dark_eyed_junco",
        640462423,
        351001488,
        "russnamitz",
        44508,
        "63276eb90201f890b977a54b95bf1d65592eef98089dbb73f2e27e66aaf49a65",
        "jpg",
    ),
    (
        "dark_eyed_junco-46",
        "dark_eyed_junco",
        649025513,
        355899297,
        "rocksand2134",
        233215,
        "2f2545fa5e7901c0383578dea84131eb4be551b8f2bd55d9f308ebc838144f39",
        "jpg",
    ),
    (
        "dark_eyed_junco-47",
        "dark_eyed_junco",
        625101518,
        343192310,
        "jtdavis05",
        106387,
        "4642076ebc5e9bcde85dbba65eb16a49d3fabe6c82cc707b4863ad4b7a4531ae",
        "jpg",
    ),
    (
        "dark_eyed_junco-48",
        "dark_eyed_junco",
        509902416,
        283764048,
        "loarie",
        271588,
        "0c1e3897caca8ff3ff4f3eff7548e30ca50dfe2c2c972106303f9e21c259dbda",
        "jpg",
    ),
    (
        "dark_eyed_junco-49",
        "dark_eyed_junco",
        627787858,
        344552039,
        "margohj",
        54717,
        "a41e557077525b064fa16a7c0b140d608d323663519cfbf424f4454d79951244",
        "jpg",
    ),
    (
        "dark_eyed_junco-50",
        "dark_eyed_junco",
        630444682,
        345899876,
        "robinlanark",
        102578,
        "aa4ffd5dbd553d60ba687d57b21b52ce21572d3c711471abbc5e959bd1126da2",
        "jpg",
    ),
    (
        "dark_eyed_junco-51",
        "dark_eyed_junco",
        635784743,
        348597723,
        "lina47741",
        317439,
        "392d1114b7fc1bb3b041d9895ceb049432a96951764e8949d5939315b96bcc54",
        "jpg",
    ),
    (
        "dark_eyed_junco-52",
        "dark_eyed_junco",
        518407567,
        288278997,
        "zzphantom",
        231733,
        "de2a5be329b583911babe0758c9ce8c0982191f743b8082eb826623256709f71",
        "jpg",
    ),
    (
        "dark_eyed_junco-53",
        "dark_eyed_junco",
        620711072,
        341026776,
        "ethologist",
        186675,
        "13f47961b2a1654ec40f6d1a2005afbf34c75060987fbb662001e66e10a023ed",
        "jpg",
    ),
    (
        "dark_eyed_junco-54",
        "dark_eyed_junco",
        638143588,
        349800424,
        "tom_lazar",
        157629,
        "0be0b84bbc956c616ce35e9e704e8c9e01f0d24db7eeb157fcaa3b3952bd9b23",
        "jpg",
    ),
    (
        "dark_eyed_junco-55",
        "dark_eyed_junco",
        668612910,
        366183670,
        "gendereuphorbia",
        76372,
        "4bfa9f780e9fd0327afec60a8b42e3378b8a9917a0461fd77a6df73b1a94e6bb",
        "jpg",
    ),
    (
        "dark_eyed_junco-56",
        "dark_eyed_junco",
        116078128,
        71308227,
        "msieges",
        47042,
        "ecd4ade5abf058ac887fabe2706eb021ada3b9ab083d10728c9b1dc26de56bbc",
        "jpeg",
    ),
    (
        "dark_eyed_junco-57",
        "dark_eyed_junco",
        117580446,
        72145931,
        "hewwowhy",
        67000,
        "1a8ca3be627c192459440c2eb07bd85b348f0e876a7de32bfc6fe0fb2959a618",
        "jpg",
    ),
    (
        "dark_eyed_junco-58",
        "dark_eyed_junco",
        117044036,
        71848977,
        "radrat",
        81035,
        "7362dd0a27d11b18d5a79abcce9a80304dc80dc0ea1fa84265a1cf0638687641",
        "jpeg",
    ),
    (
        "dark_eyed_junco-59",
        "dark_eyed_junco",
        11110817,
        8359397,
        "giselle9",
        71741,
        "b2e74f27ddca8d1e6dd4807251b07867cbfec8b6bf51d93ef7f142109f705520",
        "jpeg",
    ),
    (
        "house_finch-00",
        "house_finch",
        697940852,
        381438133,
        "ben142",
        196735,
        "a89f8e0263fdabb404b462acaa592f5dd2ac88ee4615da444470de4a1fae82d5",
        "jpg",
    ),
    (
        "house_finch-01",
        "house_finch",
        176982307,
        105476125,
        "vicki936",
        22211,
        "c377fb361df0324c7a856d9344968886ece3b94bd67188c9325b8d2d284d3a2f",
        "jpeg",
    ),
    (
        "house_finch-02",
        "house_finch",
        117990649,
        72375345,
        "kristen163",
        75945,
        "eeafad0dd2e91ecfe45c9d1f27dd0392a01bd81099549c60fd2e36a4b4342a9f",
        "jpeg",
    ),
    (
        "house_finch-03",
        "house_finch",
        389479656,
        220010434,
        "aster-asti",
        82128,
        "a8848197b4e7890e07538d492480c4275b75d04e10c1ae95aee91aaafe3319c5",
        "jpg",
    ),
    (
        "house_finch-04",
        "house_finch",
        98576538,
        61594129,
        "enspring",
        46784,
        "8b355426d8fe6327f202c6a9458cee1b95de445bfbf7e48a4b5a2c7d0eb78a8c",
        "jpg",
    ),
    (
        "house_finch-05",
        "house_finch",
        72470599,
        45698380,
        "henrya",
        61724,
        "1b96d37a7078e1b725b80af4b10848da58b0d0c17a70c8ac01e326c0a749ee6b",
        "jpeg",
    ),
    (
        "house_finch-06",
        "house_finch",
        80751781,
        50842166,
        "leahmfulton",
        44269,
        "6d6202de26f042d83ee6c5af550cd74e6ab10eb796eed2f78c83ac9e2368e4b7",
        "jpg",
    ),
    (
        "house_finch-07",
        "house_finch",
        214612538,
        126483167,
        "hamiltonturner",
        124116,
        "6b7687640c4641b974865da04cf9eaf1f86b774ebc19678f2fc39e55c8648930",
        "jpeg",
    ),
    (
        "house_finch-08",
        "house_finch",
        630196420,
        345777550,
        "truthseqr",
        149455,
        "abb84d1e327dd82c07cbea3dd5583c07453e69b2cc220397b101e597da81bd6c",
        "jpg",
    ),
    (
        "house_finch-09",
        "house_finch",
        213077180,
        125637342,
        "jnicat",
        25823,
        "e1e3baff8d0bd72339e3e49089f7f4f2c1dd383ff005e49a964a1bacc4f8ebb8",
        "jpeg",
    ),
    (
        "house_finch-10",
        "house_finch",
        500744872,
        278868417,
        "pbaff",
        149626,
        "2684cfb1fc40d7766610a5920ead0ad0c27c338ccb4fb9ed18baddda150568b0",
        "jpeg",
    ),
    (
        "house_finch-11",
        "house_finch",
        268678834,
        155431721,
        "stevestevens",
        120325,
        "5d1e8c097c214d14ef7b895bf95769ae9bc25fa799a98bbfa6f12f2f84e6af79",
        "jpeg",
    ),
    (
        "house_finch-12",
        "house_finch",
        358373136,
        202884575,
        "kcthetc1",
        52329,
        "b9929163e4fdadef26c753437ac7af3ad550aa047b15cba131c05d4d335799fe",
        "jpeg",
    ),
    (
        "house_finch-13",
        "house_finch",
        104227663,
        64793290,
        "verdantpulsar",
        101003,
        "90fe37c477bad9ed30ab119e1a7445ffc2720e548a22bb65d66b2ff7b82337d5",
        "jpeg",
    ),
    (
        "house_finch-14",
        "house_finch",
        196156834,
        116218899,
        "kgarrett",
        56801,
        "ce03d1d70a89b5e6e0088f4307f8077573c3571b91997725ca2e6c69be80e2d0",
        "jpeg",
    ),
    (
        "house_finch-15",
        "house_finch",
        454332148,
        253709711,
        "rlaortiz",
        149985,
        "cf007ef8ac57bc0eb085c9eecf9fc95eb69a29df706998de237df24f9491c61d",
        "jpeg",
    ),
    (
        "house_finch-16",
        "house_finch",
        665287910,
        295246528,
        "dinomariobob",
        74972,
        "5f6121c1f8dbfaccbb61c279a579c22600744eb4118c2eeef09e69c86f6e1a49",
        "jpg",
    ),
    (
        "house_finch-17",
        "house_finch",
        509969159,
        283798927,
        "damienxw",
        104277,
        "56d3656be473c362f1ccd09d15e62cbcfe9137d83bef7a3312d72444921c7805",
        "jpg",
    ),
    (
        "house_finch-18",
        "house_finch",
        168402062,
        100871757,
        "k-simpkins",
        86922,
        "209a884cf7618d0b85679ae3a72f237a6d03a5a3096f6ab1b2e5503c38c50d9d",
        "jpg",
    ),
    (
        "house_finch-19",
        "house_finch",
        661042217,
        362189851,
        "dougbrown",
        56426,
        "ad704994fda99779aa340ca1c637b1371f0513df6f36973263bb9ee88cf6713b",
        "jpg",
    ),
    (
        "house_finch-20",
        "house_finch",
        701862058,
        383472210,
        "dblanco",
        25222,
        "8562ed9c4e889b34f83b40c55dd93b821a6b5829a3b1c681374a53c3ca9bf01f",
        "jpg",
    ),
    (
        "house_finch-21",
        "house_finch",
        709783562,
        387605980,
        "fraskar",
        99555,
        "60a1926358c4ec04aeec9414381177600580c8237899c852e03472e53cf87f31",
        "jpg",
    ),
    (
        "house_finch-22",
        "house_finch",
        703924789,
        384540425,
        "meshhy",
        110802,
        "2df38d1aaab2c268c7a47f05e55bef1e83690eeeb91f34874c5b479a5ff0a2cb",
        "jpg",
    ),
    (
        "house_finch-23",
        "house_finch",
        705516095,
        385380385,
        "m_aniket",
        77797,
        "ba009764b66ed89efc9b0aaacd9565830c65039967961488d774ce3c9add5d64",
        "jpg",
    ),
    (
        "house_finch-24",
        "house_finch",
        577527110,
        319485540,
        "glmory",
        94407,
        "6d4cd80f8ef77f41575befae0d8534e9bc397685dceb7f629b5cbc24447a3613",
        "jpg",
    ),
    (
        "house_finch-25",
        "house_finch",
        570686433,
        315887836,
        "emilyheaton",
        229263,
        "27e0bd21c1b322aa5d74c5504dbf27bfb5a0fdeea9718b6ffcee7926f2cb2733",
        "jpg",
    ),
    (
        "house_finch-26",
        "house_finch",
        266259557,
        154114419,
        "andywilson",
        47792,
        "b0dc2de7134cc5af50b1482681b450d32b1f58765990d5d553c4838ae7936db0",
        "jpeg",
    ),
    (
        "house_finch-27",
        "house_finch",
        269729570,
        156011712,
        "kbkash",
        59048,
        "3af407624c46ba4218d24a198ed22b812d4967380bd0e7f67e960f7fbb2defe7",
        "jpeg",
    ),
    (
        "house_finch-28",
        "house_finch",
        270277531,
        156183453,
        "aparrot1",
        83424,
        "93c92a1bb7cf89e7106b0c3df65ce030c82bccade181d661952e940ddcc6282a",
        "jpg",
    ),
    (
        "house_finch-29",
        "house_finch",
        658061500,
        360629741,
        "mike_cove",
        107594,
        "deaaefaf712059b6c8a929a83e1ffd0295e12f0f291bc9869011703eecfcf372",
        "jpg",
    ),
    (
        "house_finch-30",
        "house_finch",
        545518752,
        302693775,
        "cvharris",
        208917,
        "fc5ffdbd30bc76fa4371a99f59543c9137411ddac0a691e2e8122ae7796e038b",
        "jpg",
    ),
    (
        "house_finch-31",
        "house_finch",
        662587006,
        362998939,
        "curran",
        113662,
        "644931afff18c942930717ac3ca563b63c18a0c78ab12ff88557aeede18ead63",
        "jpg",
    ),
    (
        "house_finch-32",
        "house_finch",
        665549166,
        364566546,
        "sealgyu",
        130087,
        "e6fb3e42979ce2709433e8141c4c665a70c4942166153578c7f0173328ffb2ad",
        "jpg",
    ),
    (
        "house_finch-33",
        "house_finch",
        566433127,
        313653086,
        "fake_id",
        150986,
        "cb8a50f48f65795850f57c4d8f84e49debc3deed59b6a60d5bee8ef95299650f",
        "jpg",
    ),
    (
        "house_finch-34",
        "house_finch",
        404592864,
        228025269,
        "logan_artz",
        78971,
        "d0cf631b138d199951d96538903eeb1b56e3b8da3f28c7b042d8868060f27a44",
        "jpeg",
    ),
    (
        "house_finch-35",
        "house_finch",
        423118612,
        237726292,
        "conhawn",
        153347,
        "826ddd6fa8fad5bd1450c0c7beb77fa3261f41a0afa61d7681b7096bc4225c40",
        "jpeg",
    ),
    (
        "house_finch-36",
        "house_finch",
        290080883,
        167451466,
        "vijaybarve",
        89129,
        "f0b32b4af4bdf0bc7eaf298b37a9be6dbcb175d9f72553efc239b6afd2746a69",
        "jpeg",
    ),
    (
        "house_finch-37",
        "house_finch",
        296347840,
        170825854,
        "susanaber",
        96563,
        "47832989fc212f19631e55b63c23155817209c40c66af49e4e03e42863243770",
        "jpeg",
    ),
    (
        "house_finch-38",
        "house_finch",
        414288752,
        233136904,
        "wafflemaster135",
        29548,
        "9f2039a678a69e8f8f383d43f80e3f4e6fcc81d5b99285a0e5b5e54834cf9125",
        "jpeg",
    ),
    (
        "house_finch-39",
        "house_finch",
        284751592,
        164572867,
        "efalquet",
        80806,
        "b50e0a0a4a0bae79b5a8cd034645252044601a29b081f0842443848e802e82d2",
        "jpeg",
    ),
    (
        "house_finch-40",
        "house_finch",
        205914234,
        121669780,
        "darylnolan",
        44763,
        "c04e8d7c002ad0038dcb62843e776ffb8fc86f425c2399442dd1653bbaa4b904",
        "jpeg",
    ),
    (
        "house_finch-41",
        "house_finch",
        96688995,
        60467105,
        "erikschiff",
        145431,
        "dd056b1c0d89dcb9fa06b1c38936142e5c825eb2aeed0d7d2265dc7e147be792",
        "jpg",
    ),
    (
        "house_finch-42",
        "house_finch",
        732358790,
        399184296,
        "katrinamccollough",
        41331,
        "76c899fd9b9aa84ad41888c71a36ab99431fe7201dd417f2c19a66f77388420d",
        "jpg",
    ),
    (
        "house_finch-43",
        "house_finch",
        83824778,
        52720178,
        "beesbirdsbugs",
        48816,
        "72731eb63fa9c5058f9911d11be929af9a58cfea805159d54010e9095948c39a",
        "jpg",
    ),
    (
        "house_finch-44",
        "house_finch",
        349400403,
        198261102,
        "paulgraham",
        105965,
        "b0d1f556ee4926e6a6f87201f6b73955dd9c530788e0ce57d5e09ced74ede001",
        "jpeg",
    ),
    (
        "house_finch-45",
        "house_finch",
        484652323,
        269544283,
        "robinlanark",
        29863,
        "a1d6e8ffbcef10a83a2d3674c7228af89f8d39297f44cd569d1e031da02bc516",
        "jpeg",
    ),
    (
        "house_finch-46",
        "house_finch",
        341357046,
        194221751,
        "haconra",
        80643,
        "8e11937a5c6d5a47da47f562d52d4089d58ad3f2adefadc287b30c562c7ece85",
        "jpg",
    ),
    (
        "house_finch-47",
        "house_finch",
        622140195,
        341714166,
        "rachel141",
        133938,
        "aa776ef9fd3022b6b3db824a762bdde82eeb95518a8764763bd8199683bf1d04",
        "jpg",
    ),
    (
        "house_finch-48",
        "house_finch",
        515309527,
        286632225,
        "solunasilver",
        29507,
        "72a6b495ed99de80504c6b1a86ac1fbfc9e5736bc6f43274bd6093cc91f9ea05",
        "jpg",
    ),
    (
        "house_finch-49",
        "house_finch",
        514877896,
        286416757,
        "jubileej",
        153667,
        "26cd596525a244d2ba82ec2d575834c15671080814400cf7862f614c0db68d07",
        "jpg",
    ),
    (
        "house_finch-50",
        "house_finch",
        125605349,
        76831730,
        "sarahangulo",
        60174,
        "236eb345f1cf1ab4c82fb3b2cc48b948e614f8ddfd958c87a62ce5b44a04f080",
        "jpeg",
    ),
    (
        "house_finch-51",
        "house_finch",
        12881677,
        9467101,
        "artemis224",
        173956,
        "91b5f9f66e8b84cd96169894024ebc3519de6f737d3a32ee3b70100c22b039b9",
        "jpg",
    ),
    (
        "house_finch-52",
        "house_finch",
        124405788,
        76034586,
        "radrat",
        36008,
        "bb9b3fe0b37a1ab6342b1e573f3faf3a287669b70fa9f00b5bfce83de4a376c6",
        "jpeg",
    ),
    (
        "house_finch-53",
        "house_finch",
        17468099,
        12163814,
        "andy71",
        199644,
        "6bc02aaf56c1549dd78d6711f9121d48c33fbd76a8fb623895369f05604ad30d",
        "jpg",
    ),
    (
        "house_finch-54",
        "house_finch",
        134699271,
        82117365,
        "seanwashington1",
        35334,
        "d6e0fe4a8cc690fd6369deff8ea67f4810e1f1297a6f1b9f125aae905742db2a",
        "jpeg",
    ),
    (
        "house_finch-55",
        "house_finch",
        134287410,
        81854554,
        "omcelroy",
        90394,
        "6b4a5ea4c8aa1722332ccdf420af0ca0906cfdb87348e9f6c0904d5a92d8c203",
        "jpg",
    ),
    (
        "house_finch-56",
        "house_finch",
        17763955,
        12337763,
        "jennifer510",
        167368,
        "35355c603049fdf21fc3bf1fed827b50740cfdba134542707026bb4e1f32149b",
        "jpeg",
    ),
    (
        "house_finch-57",
        "house_finch",
        18946521,
        13021113,
        "deejay",
        60576,
        "2420363f6b02433f4b03f3323148ed5e4ac9d37c58a27e859f8e48354db866c8",
        "jpeg",
    ),
    (
        "house_finch-58",
        "house_finch",
        14590841,
        10492098,
        "silvercat",
        120515,
        "23934e0376c70dd6d5ce5e9a24c4176291c877432d25257f07e27694495c879a",
        "jpeg",
    ),
    (
        "house_finch-59",
        "house_finch",
        124702263,
        76228853,
        "janeyair",
        162431,
        "5c86d7cbd28e09d44814735f1e0a21e8266bd4e6a4396b047d092d2f5f3f2111",
        "jpeg",
    ),
    (
        "american_goldfinch-00",
        "american_goldfinch",
        84579952,
        53187208,
        "glennberry",
        59673,
        "72d36079e592e0a83c2f774f9073bfd4cc81253452c925d1673217ddd4b52a36",
        "jpeg",
    ),
    (
        "american_goldfinch-01",
        "american_goldfinch",
        12533322,
        9255418,
        "braincellsgone",
        55661,
        "6344e0125e74791f43ac6e07e5e1b9fbfce6d19bc62b6bb5d83b3caff9f7bcbc",
        "jpg",
    ),
    (
        "american_goldfinch-02",
        "american_goldfinch",
        131102823,
        80016788,
        "radrat",
        98990,
        "232a944f7e3351d4916a12ef2f6d598e7b007caaa95a9b064a14c1ba3af2a6ff",
        "jpeg",
    ),
    (
        "american_goldfinch-03",
        "american_goldfinch",
        175048222,
        104466897,
        "eug302",
        44231,
        "11c723482cc75fcc3a723ac1c0818a68e2fcf74c4ca684cf60195bf0d33f274e",
        "jpg",
    ),
    (
        "american_goldfinch-04",
        "american_goldfinch",
        68849595,
        43390778,
        "mefisher",
        154503,
        "66f07bc59bb3fdedd65a4537ebabd0cafd457826b8bf4bb633181f584a3edfd1",
        "jpg",
    ),
    (
        "american_goldfinch-05",
        "american_goldfinch",
        431916465,
        242278180,
        "k-simpkins",
        45413,
        "d76e7adf33e3a84ebec24dde5438965e62ebec8595755453973846340e4f460d",
        "jpg",
    ),
    (
        "american_goldfinch-06",
        "american_goldfinch",
        230801384,
        135330401,
        "enspring",
        42886,
        "78aebfb9b28c3e16dd9618a0e1ae06df67bfa4b8550c0879427d96915c475fed",
        "jpeg",
    ),
    (
        "american_goldfinch-07",
        "american_goldfinch",
        377136648,
        213398931,
        "nathan1177",
        66516,
        "7ad75838fdf2020a8e426e97507c7dd4355da93e6eece241128c28adbe302438",
        "jpg",
    ),
    (
        "american_goldfinch-08",
        "american_goldfinch",
        294667312,
        169935316,
        "dande",
        163470,
        "39e7892e81eeef6af4887e61bc0998e17797688eb6394fcc8c9438391e875ee0",
        "jpeg",
    ),
    (
        "american_goldfinch-09",
        "american_goldfinch",
        660472044,
        361884286,
        "ben142",
        270599,
        "733d64cd50c61334682f0862f5c7859ded34cae5776ee0bc6594fee400dc4876",
        "jpg",
    ),
    (
        "american_goldfinch-10",
        "american_goldfinch",
        143215717,
        86889530,
        "memoosborne",
        129438,
        "ef4a9a771cef9ba3c2c047eb106a6aa220236dd6aaa6aade5f4ef3a37c8abcba",
        "jpeg",
    ),
    (
        "american_goldfinch-11",
        "american_goldfinch",
        403873164,
        227647158,
        "drew_baxter",
        106009,
        "74e34c776f1b9a5d375a7dfae0e309cd42dbd133b82a1ecad4a49111ac7eed49",
        "jpeg",
    ),
    (
        "american_goldfinch-12",
        "american_goldfinch",
        481153019,
        267722510,
        "vicki936",
        255397,
        "3326ddbee3270241b681cb636466e742491f110e9841f453f5158864aadaea36",
        "jpg",
    ),
    (
        "american_goldfinch-13",
        "american_goldfinch",
        213311122,
        125763980,
        "hickl",
        24740,
        "0c295b3761bced4215519ff24a4f734773b54be98de456a4444d0819456af7dd",
        "jpeg",
    ),
    (
        "american_goldfinch-14",
        "american_goldfinch",
        417352824,
        234736320,
        "joy4birds",
        81109,
        "357014c108519543471b94f39591667d1a67d87fd1fdc4702a3a449235d6bddd",
        "jpeg",
    ),
    (
        "american_goldfinch-15",
        "american_goldfinch",
        72130720,
        45486482,
        "dctphoto",
        178111,
        "17b2f3599e20161acc17fd63bf61e9f40488b9c4d5bfaefc8cae54de42997509",
        "jpeg",
    ),
    (
        "american_goldfinch-16",
        "american_goldfinch",
        45148898,
        28952026,
        "megachile",
        49358,
        "6bbc6ca0ac074e486c20ce4c3fad5dc863cb2e908efc2b1ef97494403a351252",
        "jpeg",
    ),
    (
        "american_goldfinch-17",
        "american_goldfinch",
        133251099,
        81250513,
        "raffib128",
        128110,
        "863c76587fe1de80a84b97c2e72f38ae1cf4fa972789e006d9b0b544d696c6ef",
        "jpeg",
    ),
    (
        "american_goldfinch-18",
        "american_goldfinch",
        155797129,
        93957328,
        "nathanael15",
        74086,
        "7059ae3bdeeb48fe949e5b70b788bd28c96aecd0fe49cb7be22db9a8697bf5a5",
        "jpg",
    ),
    (
        "american_goldfinch-19",
        "american_goldfinch",
        11400438,
        8535447,
        "akneidel",
        26423,
        "4af74c1d04ddc7bbb7bb0e9eb2977a1daf48dd9b9477d71d69a7c4cb5d4f785b",
        "jpg",
    ),
    (
        "american_goldfinch-20",
        "american_goldfinch",
        55990791,
        35505213,
        "conhawn",
        84915,
        "69f768c39a2180440bdcbfc6addc5d426341e8080d4cf9ba8241d57564b3e6fb",
        "jpg",
    ),
    (
        "american_goldfinch-21",
        "american_goldfinch",
        145464774,
        88186208,
        "wildreturn",
        68204,
        "565d2a3b6d404e9ea0c24737ebcc5a3a82057b1452b4790df5e9552e8c50e592",
        "jpg",
    ),
    (
        "american_goldfinch-22",
        "american_goldfinch",
        637247336,
        289067166,
        "dinomariobob",
        142312,
        "01c63d305fce8edcc3a494f0543154c6bd6aa52368e02be84cf6c23e95b94cdd",
        "jpg",
    ),
    (
        "american_goldfinch-23",
        "american_goldfinch",
        460988055,
        257029007,
        "eric112",
        60314,
        "0db1b5f0e32d1faf861a437a20794fa18e80ea8966313933c145451c9319e2a9",
        "jpg",
    ),
    (
        "american_goldfinch-24",
        "american_goldfinch",
        59072170,
        37280587,
        "bradenjudson",
        23468,
        "827d77bf9ca9cc456e867434a07b68cdceb4a20e09eb9e6b67757c80cd31ff42",
        "jpeg",
    ),
    (
        "american_goldfinch-25",
        "american_goldfinch",
        66515260,
        41915252,
        "reuvenm",
        58833,
        "90f0b687d9626fdf5d0111b794cf960cf18f3de4c8eab601fcb23c41b45c4df7",
        "jpeg",
    ),
    (
        "american_goldfinch-26",
        "american_goldfinch",
        109875643,
        67928198,
        "artemis224",
        217909,
        "99e1d7d30eb19d47c7974e9ddd7efe4d06329c952a41e15d0a4b2095b747df61",
        "jpg",
    ),
    (
        "american_goldfinch-27",
        "american_goldfinch",
        219478221,
        129171341,
        "cgmayers",
        96993,
        "93bdee27bb526b60e0edfa518a923b3c10c41ba57221d6a23f4de15bf4a91600",
        "jpeg",
    ),
    (
        "american_goldfinch-28",
        "american_goldfinch",
        99106353,
        61899131,
        "thyg",
        50544,
        "0272bf89b379c2689247f1dbd6f4b36d7f197898baad91d7a8ebdf1c87f88fbd",
        "jpg",
    ),
    (
        "american_goldfinch-29",
        "american_goldfinch",
        223110598,
        131133187,
        "marissa3",
        89214,
        "6a2bb601ac3a70e80e6ccb33d4a24af43128d1e7afe3e92d42e803234bdf532e",
        "jpg",
    ),
    (
        "american_goldfinch-30",
        "american_goldfinch",
        104649681,
        65037800,
        "kemper",
        113821,
        "f4430daf9fca2cf6deb6b988148a5c2e5ae382af8c5db7e5cd37b26e8fbc9461",
        "jpeg",
    ),
    (
        "american_goldfinch-31",
        "american_goldfinch",
        98669936,
        61655828,
        "cloaca_enthusiast",
        31964,
        "3ddcb2e2e6863ef08a40fae7640e05b36ccec9fab9892ac494f095c4588b746f",
        "jpg",
    ),
    (
        "american_goldfinch-32",
        "american_goldfinch",
        347798463,
        197465311,
        "aster-asti",
        116039,
        "e1236dd9850a412ba68386ffd342d180f582db187b83c1d33cb1e09dd51d0043",
        "jpg",
    ),
    (
        "american_goldfinch-33",
        "american_goldfinch",
        461261879,
        257163676,
        "erikschiff",
        40881,
        "eb0b1ea08fcee160edd008ffc386b69b7b50bb89096c44873b978072c581d826",
        "jpg",
    ),
    (
        "american_goldfinch-34",
        "american_goldfinch",
        343726970,
        195390446,
        "robinlanark",
        53508,
        "8b1f4e496755d1a3c4f567c6a48673998b8480ce7d38e7c52bbd94d7ee9901ed",
        "jpeg",
    ),
    (
        "american_goldfinch-35",
        "american_goldfinch",
        352259924,
        199722919,
        "carterdorscht",
        122243,
        "11edfdb53f892b1123df734cb7300e002171baeef21671a31411ff30d156bdc6",
        "jpeg",
    ),
    (
        "american_goldfinch-36",
        "american_goldfinch",
        148829913,
        90084462,
        "ian-wolfe",
        172110,
        "4f6a493313fc596665eb99f8e303ab0ee980b4932a9957993cd00b03b5360fa4",
        "jpg",
    ),
    (
        "american_goldfinch-37",
        "american_goldfinch",
        276388190,
        159958623,
        "dianeclark6280",
        39786,
        "284ca7c9e814fede2c5a15e427a46d7dd82fa6984c4c862c380b0dddff2f86db",
        "jpg",
    ),
    (
        "american_goldfinch-38",
        "american_goldfinch",
        409778763,
        230765968,
        "squidtk",
        210988,
        "b92ccce145f7a320f25edf92633e0c56aa92e8c83890b4a439ce4b3d669cb1a1",
        "jpeg",
    ),
    (
        "american_goldfinch-39",
        "american_goldfinch",
        382773537,
        216432059,
        "rocksand2134",
        195020,
        "ad33d43d9e5650dd441bbd9adb9cc9854a6a1bcc7088a8802a7fb4ccc1ee8b51",
        "jpeg",
    ),
    (
        "american_goldfinch-40",
        "american_goldfinch",
        290287368,
        167563678,
        "sean579",
        105190,
        "8703f245d4b98476f67775ca0c19b7abd7c5edf756c50b2ff502b3d058c08014",
        "jpeg",
    ),
    (
        "american_goldfinch-41",
        "american_goldfinch",
        423112648,
        237722733,
        "andrew2285",
        55719,
        "10709f5df3f4c509afd260ba51c0db5cee89c7c9076c74e9911e9613303d7754",
        "jpeg",
    ),
    (
        "american_goldfinch-42",
        "american_goldfinch",
        422361237,
        237335568,
        "wildaboutwildlife",
        34855,
        "55b0962389b589037e03378472e65ffa0e7ed0fb7e227a7441106a009f093924",
        "jpg",
    ),
    (
        "american_goldfinch-43",
        "american_goldfinch",
        291016386,
        167956837,
        "samallonthesciencemon",
        128110,
        "489934bec3eb5682067cca11cb210c171198e616bb4920753b48bb0e1480bc32",
        "jpg",
    ),
    (
        "american_goldfinch-44",
        "american_goldfinch",
        419250127,
        235728207,
        "paulgraham",
        39211,
        "be1f9ae51a6c4ee92a543ecfe1654698581d7647a3a0b46859a7cf054d8b0fd9",
        "jpeg",
    ),
    (
        "american_goldfinch-45",
        "american_goldfinch",
        416004544,
        234039610,
        "randv",
        105453,
        "942a87723af8aedc86459479437bc474817f19429806f36d8e1236538448e24f",
        "jpeg",
    ),
    (
        "american_goldfinch-46",
        "american_goldfinch",
        416408092,
        234251817,
        "kmayner",
        143385,
        "55259e21e667b88864e2b88ba8dacc8ac5be0d2606699be35c46d2b17fe67fb2",
        "jpeg",
    ),
    (
        "american_goldfinch-47",
        "american_goldfinch",
        116188790,
        71368917,
        "leahmfulton",
        48440,
        "ae01ef8ba2fb714253543a5e773e4e047e95448f4ff4d1e51bc889a20b14e1bc",
        "jpg",
    ),
    (
        "american_goldfinch-48",
        "american_goldfinch",
        118530525,
        72675993,
        "seanwashington1",
        38505,
        "d688edde61a079907cdf5011a32914432f8f7f35c12d355e308aae29b1988738",
        "jpeg",
    ),
    (
        "american_goldfinch-49",
        "american_goldfinch",
        121016576,
        74071097,
        "geenance",
        110180,
        "fb4d1fef09311949043b3c9edc0d3a85d4c20386346093bb6ea39e1b95f756b9",
        "jpeg",
    ),
    (
        "american_goldfinch-50",
        "american_goldfinch",
        3279608,
        2872951,
        "joed",
        115964,
        "c2f1463bfef4c9b157f7e3ebd33fa37570592a4cdfa276f2b7c673f543cbc3ca",
        "jpeg",
    ),
    (
        "american_goldfinch-51",
        "american_goldfinch",
        118140282,
        72458899,
        "melaniegaddy",
        243713,
        "a3d70061032ac8c665f3d17a9118c8e592d53a3669182b23b502e3573a80d8f8",
        "jpg",
    ),
    (
        "american_goldfinch-52",
        "american_goldfinch",
        7669539,
        6125334,
        "cuihenggang",
        61287,
        "36d9570131caf33f3b5cfa6ab99a26c417e7cd8e38d0c1351d1a2938bb6e9bf9",
        "jpeg",
    ),
    (
        "american_goldfinch-53",
        "american_goldfinch",
        9258865,
        7182023,
        "fake_id",
        151987,
        "131d2a52097c9ab18fad318b62e8213c1bd33acddc72065ec0e404c76220c5ee",
        "jpeg",
    ),
    (
        "american_goldfinch-54",
        "american_goldfinch",
        12222432,
        9054511,
        "myacadianforest",
        107758,
        "69b1b5274d74bb28601afc2758e57a88f8d63743b5410c70284a9f4877c721fb",
        "jpeg",
    ),
    (
        "american_goldfinch-55",
        "american_goldfinch",
        128301260,
        78400716,
        "natepow",
        67416,
        "0c988080a47484802806ef14e0d1fdb9205680eacecedbe24f8bbac2aa2873f9",
        "jpg",
    ),
    (
        "american_goldfinch-56",
        "american_goldfinch",
        129842959,
        79296314,
        "nartb",
        117002,
        "f23b2e68f4c10e2679f0a9fe2f34a894bd2deea9c90c2143462e0aa5a83a3daa",
        "jpg",
    ),
    (
        "american_goldfinch-57",
        "american_goldfinch",
        78410307,
        49351195,
        "chrisleearm",
        108066,
        "8a4b273f1a1bd548c48abcf57b86ed35028cbfb949720180a43f24976a26ac8e",
        "jpg",
    ),
    (
        "american_goldfinch-58",
        "american_goldfinch",
        72300928,
        45598176,
        "henrya",
        68593,
        "a0c2869c330aaea08fba3bfeae99392c5588980b1ec21770c220c744125c3b42",
        "jpeg",
    ),
    (
        "american_goldfinch-59",
        "american_goldfinch",
        194724999,
        115367030,
        "vlatassa",
        159105,
        "c456a23238f8f81f93df7919b92dd3fa0123fa16a875c22ec1f9d0ae6ef17324",
        "jpeg",
    ),
)

# Fresh test (review R88-M4, owner ruling 2026-10-17): 24 photographs per species that no fleet model has scored,
# drawn by tools/draw_fresh_test.py (seeded, documented iNaturalist query; record docs/fresh-test-draw.json):
# CC0, research grade, one per observer, observers disjoint from the 360 above and from every other fleet pin,
# photo and observation ids in no fleet pin, not same-place-same-day with a pinned photo, near-duplicates screened.
# Same layout as SAMPLE_RECORDS. Scored only under the pre-registered settings (score-once rule).
FRESH_TEST_RECORDS: tuple[tuple[str, str, int, int, str, int, str, str], ...] = (
    ('song_sparrow-fresh-00', 'song_sparrow', 608463067, 334988631, 'luciandavis', 75180, 'c04b6ba0bfe9a7a58d833c29c870a9351173e19b17bbb6c4d7afd8902e9d3bf2', 'jpg'),
    ('song_sparrow-fresh-01', 'song_sparrow', 182179338, 108226956, 'hllrnnr', 69464, '834c67222b9cf39c98cbaf0fd12cf53d9688a9ef7742173bd2a1729b2f57b436', 'jpg'),
    ('song_sparrow-fresh-02', 'song_sparrow', 93655243, 58652547, 'scurfytwigles', 41021, '59e98f6ddd5943a343a1606c7ad96274aeeb361810982c7329b46145585e721b', 'jpeg'),
    ('song_sparrow-fresh-03', 'song_sparrow', 612436421, 336957018, 'sapien', 162174, '2c4bc930a48b75da007b2c68186c5a06e9cb7d36c4f89646e3766cdec2271ba4', 'jpg'),
    ('song_sparrow-fresh-04', 'song_sparrow', 99117435, 61905636, 'elspethpierce', 202377, 'cb06d7861f8580f491cb30b3b409154f807ebb54eb74373803a933d05ccd56ba', 'jpeg'),
    ('song_sparrow-fresh-05', 'song_sparrow', 707388754, 386361697, 'sandiegonudi', 61432, 'ca085657b752481fd74067b552f66b4baa56450d0f7395da26947c8ff2757be5', 'jpg'),
    ('song_sparrow-fresh-06', 'song_sparrow', 18581022, 12812516, 'elinpierce', 82679, '6df4850cf13474637fe743abe258983156e5050c4fe2e6f2cbb0104e37a5da02', 'jpg'),
    ('song_sparrow-fresh-07', 'song_sparrow', 291861722, 168412420, 'twsummer', 206504, 'f9ce508314336353a27b04c74f96216c881d930c36f46bfd8bbf824b8d8373f1', 'jpeg'),
    ('song_sparrow-fresh-08', 'song_sparrow', 138802919, 84450492, 'pcartier', 56669, '743503f17e1d1ebc809ddd0e400aef597f1536f2997d28661cbf15a0aeff54fb', 'jpeg'),
    ('song_sparrow-fresh-09', 'song_sparrow', 14660798, 10531563, 'accipiter56', 97930, 'a067e8bc51574a27a066fc1c7ac4127232c99b8513de6faad45ae4fb93356ace', 'jpeg'),
    ('song_sparrow-fresh-10', 'song_sparrow', 479116111, 266652487, 'kenkneidel', 287799, '3815e59850bb82da299414cd6752726b86f8dc8aee375a2991f0957a703284b7', 'jpg'),
    ('song_sparrow-fresh-11', 'song_sparrow', 261474881, 151535354, 'funnieanimals', 201762, '62b03726cd46b22f2e7c550157cc37d0b7310e2e2c9a01c9fc5c82c4e8e9f4ae', 'jpeg'),
    ('song_sparrow-fresh-12', 'song_sparrow', 63678842, 40093573, 'sheilsun', 112819, 'dd58d9ce2556a39423ba4f804b59c1aab3083c194a665bfd894fbf226e057a09', 'jpg'),
    ('song_sparrow-fresh-13', 'song_sparrow', 516193066, 287098105, 'buriedanimal', 208843, '7ca09f0a1c18626a0f10061284a0b94820dfbd1b35ff60fcb22d81df5edd8ec3', 'jpg'),
    ('song_sparrow-fresh-14', 'song_sparrow', 618109529, 339746213, 'jzupan', 88977, '828c1a4aceccb1274c6b26441ea8a2cf1bbf18edc5a2b1871044bb4af1d58757', 'jpg'),
    ('song_sparrow-fresh-15', 'song_sparrow', 369451012, 208795328, 'soo_moths', 142415, '84fd894c15a452cdabd11ed1c31006f819e7f2e26c509d3b8cc0373ba6ad302b', 'jpg'),
    ('song_sparrow-fresh-16', 'song_sparrow', 277765561, 160729732, 'cam_boucher', 251394, '5e2fb4d0e308e61f4e350249cecc20a4be460d7a70b5253a58b92b978f204f2c', 'jpeg'),
    ('song_sparrow-fresh-17', 'song_sparrow', 517781755, 287950343, 'najah22', 197931, '64c0a619a354202a73dff1372e571135eeda66701d4678a7fb5b0af96b57fd84', 'jpg'),
    ('song_sparrow-fresh-18', 'song_sparrow', 358908828, 203163657, 'whiteoak', 114954, 'f7b4d726de07c555d93b1d8fba7e0a073356858377b6ef9eada240dd30cc25cc', 'jpg'),
    ('song_sparrow-fresh-19', 'song_sparrow', 191930717, 113613698, 'gsexauer', 81155, '522451aa51597b22e969a126a883de0e210fef770018c5aad1bc1a9b816730cd', 'jpeg'),
    ('song_sparrow-fresh-20', 'song_sparrow', 664217525, 363858397, 'saketas', 68037, '9f0f6d17923b36ef44488a49bd9404299910d513b6183458e35ceaf19cf6af8b', 'jpg'),
    ('song_sparrow-fresh-21', 'song_sparrow', 614282177, 337859471, 'eschov39', 282810, 'b22671471432c983c89e1d36e35c6f4fec80c0bd4aa64c9fef50f2e9cb5ef63a', 'jpg'),
    ('song_sparrow-fresh-22', 'song_sparrow', 251977044, 146548301, 'envirosue', 169430, '558873336ba306576da3625fb1990c455014ec3cf0171929903cf060d34342b7', 'jpeg'),
    ('song_sparrow-fresh-23', 'song_sparrow', 224088512, 131660909, 'surlyaardvark', 151185, 'ed667077dced2fba79680da602b2759ec12ece911cfba3af220e315bd2adce0a', 'jpeg'),
    ('chipping_sparrow-fresh-00', 'chipping_sparrow', 510552876, 284107182, 'walkingfern', 58997, '15b0da9e605d43781d415f3b0f0a6ad1fd478e7bb73be5d783fd635711bb8546', 'jpg'),
    ('chipping_sparrow-fresh-01', 'chipping_sparrow', 194093535, 114975512, 'valeriedupee', 211006, 'e9762b5601d9561ef9b13b479f0655ba198330593ed7b9a2413ef280297ed5ec', 'jpeg'),
    ('chipping_sparrow-fresh-02', 'chipping_sparrow', 335187036, 191112176, 'nickajacket', 75237, '4028326c510c52b917bd358cfa44ae456b877a713885e3f441cb007d784abbad', 'jpg'),
    ('chipping_sparrow-fresh-03', 'chipping_sparrow', 37070686, 23966199, 'lotteryd', 213860, 'ff8f63a91197dcad6270763314ae564fb69ba3054488e668b94cea89266c6a66', 'jpeg'),
    ('chipping_sparrow-fresh-04', 'chipping_sparrow', 662839554, 363136470, 'con19776', 112549, '84b366fde2aaf466dc25ff3e69f8357f74a0f2c4d76bd5d31df1578b35bad61b', 'jpg'),
    ('chipping_sparrow-fresh-05', 'chipping_sparrow', 696545425, 380718370, 'jeremie_casavant', 210199, 'bb28ce4fcdd80296eae3d57531b0714591d6be9bc9730747ded75de51ed17d04', 'jpg'),
    ('chipping_sparrow-fresh-06', 'chipping_sparrow', 203031423, 120066020, 'abbybenson', 220781, '3cb07691518c5b8607de73240bee06864672faf5073c86165355e3ee03c5e545', 'jpeg'),
    ('chipping_sparrow-fresh-07', 'chipping_sparrow', 192702548, 114093275, 'toadhunter', 59486, 'd07ef2d51c4e9e1fd773b5975addcf247374ec750c95d3383ee6a5dade84f4b1', 'jpg'),
    ('chipping_sparrow-fresh-08', 'chipping_sparrow', 71453864, 45068329, 'erinpitts', 191598, 'c4c1f8dc36f95de9efc10040cca0f837fe0fb918702a8c542fccc5e65378c6f9', 'jpeg'),
    ('chipping_sparrow-fresh-09', 'chipping_sparrow', 119895759, 73440544, 'cnpsstevew', 49205, '8cde57e8f6f172223c4887f965b8f0aafc241bc7708983c911445995ed1688ba', 'jpeg'),
    ('chipping_sparrow-fresh-10', 'chipping_sparrow', 357001420, 202165785, 'alexandew2552', 209428, '4e0c23bdc5c0d2e2aa40d5464c1c4ce0c13368faf5e192711090792c610d3240', 'jpeg'),
    ('chipping_sparrow-fresh-11', 'chipping_sparrow', 5935782, 4882265, 'mmulqueen', 124824, '09ab8dc867a7b8dde44f229973798ce7ebc77d335a82ead49f2aded1df52a6d8', 'jpg'),
    ('chipping_sparrow-fresh-12', 'chipping_sparrow', 88231599, 55411019, 'james_david', 105640, 'a498ffd0c59f1b49af1b4d73743b1ea93c63ec9cf7b337355d07660758ac85a8', 'jpeg'),
    ('chipping_sparrow-fresh-13', 'chipping_sparrow', 374585711, 211937673, 'deedesie', 101650, 'fa1e8bd6ebf512266036ccbe182c9c400fb160357fdc45cc24d0753eed21b6dc', 'jpg'),
    ('chipping_sparrow-fresh-14', 'chipping_sparrow', 323140429, 185063050, 'bobhunter2', 129327, 'e4a30240aec477efa49226a64970851e7c3e7535c18f65015e2ac7fa548ad1df', 'jpg'),
    ('chipping_sparrow-fresh-15', 'chipping_sparrow', 565898699, 313375817, 'schneepschnorp', 68078, '1b1ccf5b92063a4b7f2c94476f82d9a910de4b054544ebba76e3cb3bc3e25b4b', 'jpg'),
    ('chipping_sparrow-fresh-16', 'chipping_sparrow', 501408972, 279231273, 'loganstecker', 67947, 'ae2fba81343ab225649bdfe745e2a67ecac1569849234dfb524fb718129d5666', 'jpg'),
    ('chipping_sparrow-fresh-17', 'chipping_sparrow', 668914990, 366335071, 'chris_and_pamela_pipes', 59885, '5302cf35f328e7fff23395e1ef0510d8651d4fe8f09be009b7c110595dd8a070', 'jpg'),
    ('chipping_sparrow-fresh-18', 'chipping_sparrow', 43316094, 27814624, 'jacksonkusack', 134079, '06c6ade51821d474cae3e672cbce9295f69607ec805e580e233960bcc67e863b', 'jpeg'),
    ('chipping_sparrow-fresh-19', 'chipping_sparrow', 550929273, 305551852, 'mheff449', 336359, '40467d683fab9349259f0dd97a101946db00adc1d03ea945637709fd7d964228', 'jpg'),
    ('chipping_sparrow-fresh-20', 'chipping_sparrow', 24343586, 16013039, 'hannawacker', 174262, 'e7b5c7604a55c93e1305af5454ba50e3fc5568d27fbbb48e590308d68518e746', 'jpg'),
    ('chipping_sparrow-fresh-21', 'chipping_sparrow', 617894398, 339640841, 'outdoormatt', 51773, 'deb703c85a65583e8dfa52bea574136aa3a6125a5fa0229097fc742d5c3d5ce4', 'jpg'),
    ('chipping_sparrow-fresh-22', 'chipping_sparrow', 362367434, 204975744, 'sgtbird08', 100909, '55354e3741f29481e92157e469f1f31cb7c6060f5f1f616cd27a6c16f6f8b54f', 'jpg'),
    ('chipping_sparrow-fresh-23', 'chipping_sparrow', 585193163, 323415428, 'andrewholle', 51899, '3bcf5074a8d8ff06a9d62a033aa5bcce487a6e25fba00127472634f05ec318fb', 'jpg'),
    ('white_throated_sparrow-fresh-00', 'white_throated_sparrow', 110715186, 68384201, 'laneneely', 139591, '6a35bb7869915a4878d11042acfc5a258a412e68c8c2cbd7d175945fbda5fd1c', 'jpg'),
    ('white_throated_sparrow-fresh-01', 'white_throated_sparrow', 104006229, 64672296, 'lynnharper', 84324, '47b703940e9f9bd2071a490c0e31cb2dc04337296b4aade7ba9f12008c4311b7', 'jpeg'),
    ('white_throated_sparrow-fresh-02', 'white_throated_sparrow', 467205913, 260187552, 'kdaniel20816', 127787, '86812763c7b9748584d271ad9e9c58a66cd53d25532c07e81b1312268c17c4e0', 'jpg'),
    ('white_throated_sparrow-fresh-03', 'white_throated_sparrow', 276223250, 159859844, 'hurlbert', 115598, '332d783deee14599d73f7559792c7ac48db66165096edd4e694199144e65befe', 'jpg'),
    ('white_throated_sparrow-fresh-04', 'white_throated_sparrow', 648785848, 355770581, 'swords', 48207, 'b5a07b11b0c77209f6f6518ab3cdd06dfb6a2a23ff454eee31f1e995e0ac7aca', 'jpg'),
    ('white_throated_sparrow-fresh-05', 'white_throated_sparrow', 586068257, 323860004, 'cobrien207', 167687, '3dd5725b0403c4e60fd43e476bab094ed113938ddf6bb2d88925637c9c64516a', 'jpg'),
    ('white_throated_sparrow-fresh-06', 'white_throated_sparrow', 30665250, 19885474, 'lanechaffin', 78640, 'd228189d7c44f0c14d75d8c3bbe6399389629361a37b8c6362e5bff22fda0f4a', 'jpeg'),
    ('white_throated_sparrow-fresh-07', 'white_throated_sparrow', 364939533, 206337837, 'wllam', 221504, '55b2e91d2f2bb7ff990a90e8a1605a31552a3c886ce5df42c36a0074118eeafd', 'jpeg'),
    ('white_throated_sparrow-fresh-08', 'white_throated_sparrow', 345067211, 196054461, 'beewonder', 75198, 'fa063a05ec377312784f910ebdb5f6ee6d621f17aecfe0b906c2d225914580bd', 'jpeg'),
    ('white_throated_sparrow-fresh-09', 'white_throated_sparrow', 112809085, 69514125, 'cazotea', 127260, 'ec146f5c229b70adafbc7c1708cdb0ad5ce490b26acf033e31d7b2125144d44f', 'jpg'),
    ('white_throated_sparrow-fresh-10', 'white_throated_sparrow', 644569146, 353239525, 'timfornature', 64401, '210acddc24bac2325327f84974efc60640bc4e06dce72cb8c0398b4488913e26', 'jpg'),
    ('white_throated_sparrow-fresh-11', 'white_throated_sparrow', 242766, 188642, 'kpmcfarland', 48525, 'ec022a619ff10913efa3a23688cfa8064639ac56de04b8c3456153db05045fe9', 'jpg'),
    ('white_throated_sparrow-fresh-12', 'white_throated_sparrow', 582995994, 322287406, 'wachzim', 225789, '1234445730bc31d5cb22398a3632abd127fb60839da1346f241adef40e2f6815', 'jpg'),
    ('white_throated_sparrow-fresh-13', 'white_throated_sparrow', 597446928, 329577014, 'deannawillistonofs', 232669, '1af42aa7bf026d0f99ad39e3ded0431b072a8646e913358da31b756c24c31337', 'jpg'),
    ('white_throated_sparrow-fresh-14', 'white_throated_sparrow', 240698005, 140600938, 'jannyrocks', 132464, 'adc5fc57a901b548d87a0217b51a2a988a207b301e8afb07d6b122413cf1ff61', 'jpeg'),
    ('white_throated_sparrow-fresh-15', 'white_throated_sparrow', 461329989, 257196693, 'gcasp60', 85287, 'f16be4e26a19240c66b52dc45dabb725da1bf7e3c74227963cd536292939be2b', 'jpg'),
    ('white_throated_sparrow-fresh-16', 'white_throated_sparrow', 29422399, 19160981, 'evie_lazz', 95008, 'ab1500f6f7238a4b2bcc0e742e776664ec6bcef076450e3ad78c286f68d8b668', 'jpg'),
    ('white_throated_sparrow-fresh-17', 'white_throated_sparrow', 443261296, 248116615, 'scneill', 155733, '3ffb3230454f1a1d04e32c4cc80cd4b7b7df08469e2a972c7841df5e8702c9c0', 'jpeg'),
    ('white_throated_sparrow-fresh-18', 'white_throated_sparrow', 360377165, 203928991, 'organellie', 288906, '290af3a36c513bf9cd66bf7dc7b2d0712181a89a6e5b9245574f91baf39fbba3', 'jpeg'),
    ('white_throated_sparrow-fresh-19', 'white_throated_sparrow', 460935049, 257002750, 'christinedufour', 157990, 'c69589336f06989784d10a2520aea58dbcff5269aa08ca839115dc083b51eb16', 'jpeg'),
    ('white_throated_sparrow-fresh-20', 'white_throated_sparrow', 642869089, 352225184, 'decaturjosh', 306500, 'f51ca0b054de333be776d046d7fe21726d9347f3696023b1680b155a1bc0900b', 'jpg'),
    ('white_throated_sparrow-fresh-21', 'white_throated_sparrow', 641990827, 351775398, 'jcollatz', 137997, 'ac16db40bcaba45861365c9d2e4da6d0f5afd184035867f5a5893c8d38d1ac97', 'jpg'),
    ('white_throated_sparrow-fresh-22', 'white_throated_sparrow', 556498565, 308457881, 'abeemcc', 103003, '005070a7020de65188e1baa660a79c7dafa9a9a524eefa90edf126037c9141ee', 'jpg'),
    ('white_throated_sparrow-fresh-23', 'white_throated_sparrow', 125669418, 76837504, 'davidrobert2', 72585, 'b820a2103c5fcfa5204027319a876452fd5384a95f3120ecf77369e4d7e50824', 'jpeg'),
    ('dark_eyed_junco-fresh-00', 'dark_eyed_junco', 242886646, 141738189, 'spottytowhee', 288512, 'b4eccd7172cc9083d5f1088110f6770b984ba5b27ebe5a1a146e82ad4f698c02', 'png'),
    ('dark_eyed_junco-fresh-01', 'dark_eyed_junco', 221378302, 130196109, 'qgroom', 98315, '924f734ca20aca2dd90981ca860bfe07f9e6edd08f9a408843364e2953e7a601', 'jpeg'),
    ('dark_eyed_junco-fresh-02', 'dark_eyed_junco', 61470706, 38657109, 'al_kordesch', 87240, '7c979cfc0afc0b7ba1885d1ce0c65e5d8274128fcfe1b920f3749b1dc6eba7d3', 'jpg'),
    ('dark_eyed_junco-fresh-03', 'dark_eyed_junco', 630318725, 345836427, 'hemo_noises', 207438, '3862b70cd40ca7647653b48c1e53fc366722eb91b31808b81e1fe6a6a90101ed', 'jpg'),
    ('dark_eyed_junco-fresh-04', 'dark_eyed_junco', 469945163, 261568364, 'nickjeff13', 168964, '05e71d6040737d684850596026d37d7bd7e2dbf021a9a3554683b563f74692d4', 'jpeg'),
    ('dark_eyed_junco-fresh-05', 'dark_eyed_junco', 456328748, 254726185, 'daiyadoggo', 177576, '2195820b805018672219a0004064fb1b38cbcc08639327375f03ca304a9c0ba2', 'jpeg'),
    ('dark_eyed_junco-fresh-06', 'dark_eyed_junco', 38045176, 24605631, 'charliethrift', 91326, '1d9064963916bda04d236916d08308989376b6429cb12a4073296ac7f613dc30', 'jpg'),
    ('dark_eyed_junco-fresh-07', 'dark_eyed_junco', 412410713, 232156907, 'ferferrari', 150038, 'b03b3d0f520ef048984c865cecd20a4376ee4cb4585e8068c728f5c6e1ddf278', 'jpg'),
    ('dark_eyed_junco-fresh-08', 'dark_eyed_junco', 178117618, 106071828, 'jeanhm', 40119, 'f80782d1e8374216b5c412b3de89ebcebb3f786b3044c0acd088f9c4e569b619', 'jpg'),
    ('dark_eyed_junco-fresh-09', 'dark_eyed_junco', 289042256, 166894401, 'stekkelpak', 139849, '4b46a2b5fcd623e2555f0fb424649c23495988602287f6502036d48ea1e1ef8b', 'jpg'),
    ('dark_eyed_junco-fresh-10', 'dark_eyed_junco', 662534026, 362971785, 'stephbrulot', 275484, 'd1034ee6f621524f8e9f5dc19c4e5f92a6d8731ba543f8ac01ed4a73ed7acbc2', 'jpg'),
    ('dark_eyed_junco-fresh-11', 'dark_eyed_junco', 494615731, 275230539, 'borealisbt', 74471, '104e38809e90b6b37d3664836c8cbdaad16f2fe17ca96769d9b5f1199cbd0220', 'jpeg'),
    ('dark_eyed_junco-fresh-12', 'dark_eyed_junco', 663270688, 363360500, 'enzymatic', 184486, 'c3aeb0f0e184ba204f6772c0073f3292371d188ff29a5ce4bcaf91f50ba75358', 'jpg'),
    ('dark_eyed_junco-fresh-13', 'dark_eyed_junco', 238128600, 139245836, 'savbell', 203856, '1316378ae658be50075e06f16d38422ccdca6377b8008d39d96f4b1a479239c5', 'jpeg'),
    ('dark_eyed_junco-fresh-14', 'dark_eyed_junco', 259435005, 150453383, 'catslovetofindplants', 30184, 'fbce51fa99df47162891cf36363242aa9d7466c5cd35ec893db88886e756b474', 'jpg'),
    ('dark_eyed_junco-fresh-15', 'dark_eyed_junco', 644618463, 353267460, 'moater', 98292, 'b4417124fbb5cb276c1209810993037960eabd3aff31f228a645f6360117dc69', 'jpg'),
    ('dark_eyed_junco-fresh-16', 'dark_eyed_junco', 469695035, 261442490, 'adamcvean', 222227, 'e97cfa28ec28e1ba4f97aa69ea71215fcb38a644821f8c2d9850ac713edacfc4', 'jpeg'),
    ('dark_eyed_junco-fresh-17', 'dark_eyed_junco', 333218419, 190121944, 'underscore1234', 104233, '0c676f266d873c2116a860ddf36f039977ca6e311c0940f9e2fede9599d14bad', 'jpg'),
    ('dark_eyed_junco-fresh-18', 'dark_eyed_junco', 616763094, 339085907, 'exploreremmanuel', 109748, 'bf9e2aa206c358d87a2ad5e4fa8cfd2b2b1f8fe5a4225c27f1be579ed3ad3c98', 'jpg'),
    ('dark_eyed_junco-fresh-19', 'dark_eyed_junco', 449086162, 251030419, 'iamfindingplants', 123556, '90997531430fa00bf46202fdf7b16a10b6953c8aef3c43ff452d13476355ecf3', 'jpeg'),
    ('dark_eyed_junco-fresh-20', 'dark_eyed_junco', 107523128, 66667720, 'jairusjames', 152106, '55c2b7e0317585378d3f9a1776c7ecbeeff971c04f7c9681d00b7ac3901ac7e9', 'jpeg'),
    ('dark_eyed_junco-fresh-21', 'dark_eyed_junco', 220870876, 129925331, 'rocksandy', 160164, 'a362bfb01fca3ca4c6e0de93963c00f0175199227afad56cd70ea3c2f8312a88', 'jpeg'),
    ('dark_eyed_junco-fresh-22', 'dark_eyed_junco', 369389776, 208768991, 'rangersara', 164388, 'f3c77ab1370b6bf7d7b73bcd58ec74f4216c8b39c4858cd7938cae32cae1c6fa', 'jpeg'),
    ('dark_eyed_junco-fresh-23', 'dark_eyed_junco', 538163855, 298754190, 'elacroix-carignan', 148325, '4829f3065b3c6c1aa6422751bbe619b5580fb2b3e29350e84ef71f6011853db6', 'jpg'),
    ('house_finch-fresh-00', 'house_finch', 177699983, 105851652, 'agoldber', 48248, 'dd9702f55a19cc16271f26a3f4315ba2831ebe103293411da5f78299353ffe3e', 'jpg'),
    ('house_finch-fresh-01', 'house_finch', 189065008, 111959227, 'dbugs', 89956, 'b4da7de92778a7a5c8fed3a3faaf1b87069fd36d6edc66898e033540c243f346', 'jpg'),
    ('house_finch-fresh-02', 'house_finch', 57313533, 36269585, 'tomerler', 73123, 'c87520385ff705b5b84c1c6e8adaf16a208d602b49b2925f97299a4fffe68fbb', 'jpg'),
    ('house_finch-fresh-03', 'house_finch', 484498130, 269460524, 'mdrogge', 158429, '4e728ca296aa011c7271f3bfc53be143dd64fc000af66bffd2ab3a961b4a403d', 'jpeg'),
    ('house_finch-fresh-04', 'house_finch', 626731047, 344000064, 'ddoll02', 135649, '0a1a745898be738eb4a1741bae7ff0ff639c9cce76fcbf028e9441bbc9e7e1db', 'jpg'),
    ('house_finch-fresh-05', 'house_finch', 86205515, 54185509, 'susana15', 34529, 'b89ed1e1ff726911494979be324fdb5467995ccb6332ff2b0187296bfd524ff3', 'jpg'),
    ('house_finch-fresh-06', 'house_finch', 555341959, 307854975, 'eekim', 201808, 'cf876e1340acccac02d4f9856d9981fc460edbfb08c6e46db0b04d999c21998e', 'jpg'),
    ('house_finch-fresh-07', 'house_finch', 202937826, 120013849, 'peakaytea', 139581, '7d3149825be8a37f2f464f1fcd091e650c15d2af45472fc9bd0b3d32a1024676', 'jpeg'),
    ('house_finch-fresh-08', 'house_finch', 449450070, 251216692, 'loreplease', 90746, 'd794717274a31d30fe39dc8e8b98e4092c0beefbe7a39fd2cbdba66f3a07cc92', 'jpeg'),
    ('house_finch-fresh-09', 'house_finch', 508844287, 283198384, 'gendgrr', 89637, 'fc7c0a7bf270bedcc82f21347033a1f50fc7b7d2a9287c588edefc696f9e50e3', 'jpg'),
    ('house_finch-fresh-10', 'house_finch', 638837344, 350151385, 'theyukacopter', 148149, 'aa5ee957b953875812b770317c25541465fe34cf7a18e8fa54325616aea9e442', 'jpg'),
    ('house_finch-fresh-11', 'house_finch', 71853840, 45317337, 'dtknox', 58197, '5572a3f4c2ac13b7e7dd25aee6a5e545e16418c79d99ea7328dfef2c223013c0', 'jpg'),
    ('house_finch-fresh-12', 'house_finch', 19923541, 13552267, 'quercus_e', 122502, '4e2cfc2aed379d0840fba2f2e35f95b24db83212e2503c0d57a8d486756987c3', 'jpeg'),
    ('house_finch-fresh-13', 'house_finch', 632743463, 347063021, 'bob73', 135530, '0f1bf2d143c2c278fd48048ef52b29e86b8ac6f9fc823047015ef72990f33dca', 'jpg'),
    ('house_finch-fresh-14', 'house_finch', 362746960, 205174922, 'fraxl', 147282, '0b231cc647b210dea526042c00964e8610c38382cbf66acb2d7a6bf975799d4b', 'jpeg'),
    ('house_finch-fresh-15', 'house_finch', 340346326, 193728132, 'physicscoach', 85091, '8ee2785402bca769f0221ef3dcad4fbcc706c8296ed3b082230fa5ab2a86f036', 'jpg'),
    ('house_finch-fresh-16', 'house_finch', 526182275, 292398806, 'hikenc', 140274, 'aca1ea622a1cd1dcfde318319eca1a1810de1566edf9a7ad5ac66b5971d1bcfa', 'jpg'),
    ('house_finch-fresh-17', 'house_finch', 347537273, 197337458, 'nonbinary-naturalist', 70987, 'e469b47622b1a285de8aae7bce309e5eb889ef645b6944df7b6ad1234377ef9e', 'jpeg'),
    ('house_finch-fresh-18', 'house_finch', 28672559, 18726049, 'yawwhite', 33950, 'e4bbd79b61d199c0dd10af4c06166c1c447c8455252834f19adecc9f953766a8', 'jpg'),
    ('house_finch-fresh-19', 'house_finch', 123417271, 75324193, 'vireokilledtheradiostar', 147253, '4071deaf2934532ab8d9faa2285e3d0885ab8d7172e9658c4834ad6e7bafe688', 'jpg'),
    ('house_finch-fresh-20', 'house_finch', 622574654, 341932829, 'laura_foy', 224030, 'd64e8575682bdab8ec3aebde2f123274bf8e1c2ff01a585b0970591eceacd8aa', 'jpg'),
    ('house_finch-fresh-21', 'house_finch', 480969802, 267626086, 'rainbowthedragoncat', 54522, '567519ceb6add8c207fd7c76f712a03abe21a754a69739af28e3aeba6d50447e', 'jpeg'),
    ('house_finch-fresh-22', 'house_finch', 524394830, 291454491, 'timackerman', 185625, 'c3aa7a84f8230e1bef5e2fd3046441703fa72d2c494c08cedc7116caac663dc7', 'jpg'),
    ('house_finch-fresh-23', 'house_finch', 335202125, 191119861, 'johnkenny54', 43762, '394e404945ad90c942d7c1fe00d41e305af5f88f1b8f176a989e11aa821e7a09', 'jpeg'),
    ('american_goldfinch-fresh-00', 'american_goldfinch', 573000936, 317119590, 'dregyn', 31418, '5fa7be4e61c2bff0d0d3a79b569df36f3ca65e5e32e24ed2682a63e48a6682cf', 'jpg'),
    ('american_goldfinch-fresh-01', 'american_goldfinch', 217714284, 128187752, 'lia_mckinnon', 154534, '3bff5963be9148e55193c8ac79bc3aac3ae19153b92dadd4ff7e5d6cba8c60ca', 'jpeg'),
    ('american_goldfinch-fresh-02', 'american_goldfinch', 500626040, 278804563, 'mike_dee', 210940, '6f76ca4af650eefa812b0f68e82dbd65c59b6d91a963a05685dcd2cb38064f2e', 'jpeg'),
    ('american_goldfinch-fresh-03', 'american_goldfinch', 677273161, 370719595, 'dave_logsdon', 229113, '7a03106aca97bfa3d4ac0cdf121b3b6ef7edf8e40ac6ab5254ac275bbc6f2429', 'jpg'),
    ('american_goldfinch-fresh-04', 'american_goldfinch', 18606200, 12826779, 'timotheepoisot', 136950, '9b613b0f037c9091dc20c6a5500a061998b835aeae36c31543e0a413378c6e87', 'jpeg'),
    ('american_goldfinch-fresh-05', 'american_goldfinch', 307842953, 177029452, 'beewatcher73', 71391, 'c43e1226c1beaec1fe473e2ad28b24e5d40083bff55e9997a7c57d4ab0a28b33', 'jpg'),
    ('american_goldfinch-fresh-06', 'american_goldfinch', 113118226, 69681546, 'kadenlikesmilk', 84087, '26a1f69551c1937b1a0d338ce57b9d2ea9c6d5dcaf44aab5f4f9e5c4fae8b5eb', 'jpeg'),
    ('american_goldfinch-fresh-07', 'american_goldfinch', 58668924, 37049233, 'bradyreed', 86427, '5c6f9bb47d73133b371cea6fd91c65bb48b73df24bd5bc5ca96567bfd6c860d2', 'jpg'),
    ('american_goldfinch-fresh-08', 'american_goldfinch', 139823769, 85024998, 'bcrobyn', 86342, '004e2a4da0e2ffefc491b725c401b3b4c68a2cf3fbec6f669870cb4833c900ff', 'jpg'),
    ('american_goldfinch-fresh-09', 'american_goldfinch', 592148534, 326904137, 'morekari', 338843, '810af743a2ab12cf41409037b4e2e38fdbecec96547ddcf6261bd2c30cd02b3d', 'jpg'),
    ('american_goldfinch-fresh-10', 'american_goldfinch', 506831221, 282131531, 'joshuagsmith', 23947, '718fb10a8d180d96593b97a108f24276cdd582fe3412b28201ea987b332fc109', 'jpg'),
    ('american_goldfinch-fresh-11', 'american_goldfinch', 727092182, 396482986, 'edwinwilke', 205600, '0c9a3452e110c4a67070f658ae80c7e2aa09871dea98af5763c714680c2e9fd6', 'jpg'),
    ('american_goldfinch-fresh-12', 'american_goldfinch', 44697567, 28663292, 'petsauce', 123962, '9866c302aaffb24a3c9e406dc87c8d001a2fd10dc8e2f73cdc662b11d190e5ac', 'jpeg'),
    ('american_goldfinch-fresh-13', 'american_goldfinch', 258472125, 149950365, 'izzykracken', 115551, 'd440e4c89d61779cb466e4ec58afc6e1c28e75c9da69733924ecc79c2c07099d', 'jpeg'),
    ('american_goldfinch-fresh-14', 'american_goldfinch', 679949033, 372117104, 'koolmickie', 190533, 'a247a2166d3d16ded30781e57767edd49762ab2c95a330840170fe60dba5242b', 'jpg'),
    ('american_goldfinch-fresh-15', 'american_goldfinch', 645971822, 354076044, 'huntingbon', 123543, '082d2a92adb431fa413667bc10ee307c4d49fd2b8777f82d0d320f1a016b21df', 'jpg'),
    ('american_goldfinch-fresh-16', 'american_goldfinch', 649777303, 356312161, 'matt-b', 197252, 'fd4600456b466026b72ee6380c02ef78f064e55513faf960d2449b5d7ce6d9ad', 'jpg'),
    ('american_goldfinch-fresh-17', 'american_goldfinch', 92459055, 57938331, 'sm40052', 107173, '93ed3d977f72836519ce92d7a233693fcf406e74fbfb36162927661ac6e713b8', 'jpeg'),
    ('american_goldfinch-fresh-18', 'american_goldfinch', 196889709, 116630001, 'sshehan', 47469, 'cc2f035a255baf0f33a0e13bb2db91871d027a6d581065849d476e56f6b2d383', 'jpg'),
    ('american_goldfinch-fresh-19', 'american_goldfinch', 693703698, 379248604, 'griz3', 47035, '50fd142ecaa1b243d76b8c71906676df0b009ab065ad15c4da265fbbe6053b1b', 'jpg'),
    ('american_goldfinch-fresh-20', 'american_goldfinch', 333600154, 190318002, 'bobhoard', 131854, '59c8114b4103d996cab60bc521739fbb0a8e08b4843284cd0e1fd726656501b1', 'jpeg'),
    ('american_goldfinch-fresh-21', 'american_goldfinch', 469808580, 261489181, 'anthony296', 133994, '9ea5b90998d887e9618a295c6fcdb79c2e95d41084a7cb7eb01f70ab8d405e5d', 'jpeg'),
    ('american_goldfinch-fresh-22', 'american_goldfinch', 70063264, 44191404, 'rgshaw', 109013, 'a18f7e25656c0e04de573144c0b94bdac3b1e284a06295c1c74fac636b96be57', 'jpeg'),
    ('american_goldfinch-fresh-23', 'american_goldfinch', 40277577, 25967786, 'aggie_wildlifer', 165031, '163bcb86a0c6ab8cf3f56c57f360543db24e1de106b1f6faf70f78ba810edaf0', 'jpeg'),
)
FRESH_TEST_DIGEST = "57a6730d6a3b9d3c9d250b4afa6cab2f47ee169e245a1b38ff85603cbf3f61b6"  # validate_dataset digest of the 144 fresh-test records


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def photo_url(photo_id: int, ext: str = "jpg") -> str:
    """The served object for a pinned photo; `ext` is its recorded original extension (jpg, jpeg or png,
    either case — the bucket key is case-sensitive)."""
    if ext.lower() not in ("jpg", "jpeg", "png"):
        raise ValueError(f"unsupported photo extension {ext!r}")
    return f"{CORPUS_BASE_URL}{photo_id}/medium.{ext}"


def observation_url(observation_id: int) -> str:
    return f"https://www.inaturalist.org/observations/{observation_id}"


def fetch_corpus(
    *, cache_dir: str | Path | None = None, fetcher: Any = None, records: Sequence[tuple] | None = None
) -> dict[str, bytes]:
    """Return every pinned photo (bytes keyed by record id) from the cache or the open-data bucket; a cached or
    fetched file is refused on any byte-size or SHA-256 mismatch. `records` defaults to the 360-photo sample;
    `FRESH_TEST_RECORDS` fetches the fresh test."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for rid, _label, photo_id, _obs, _user, size, digest, ext in SAMPLE_RECORDS if records is None else records:
        local = cache / f"{photo_id}.jpg"
        data = local.read_bytes() if local.is_file() else b""
        if len(data) != size or _sha256_bytes(data) != digest:
            url = photo_url(photo_id, ext)
            if fetcher is not None:
                data = fetcher(url)
            else:
                request = urllib.request.Request(url, headers={"User-Agent": "swin2sr-super-resolution-pipeline"})
                with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - pinned https URL
                    data = response.read()
            if len(data) != size:
                raise ValueError(f"{rid}: photo {photo_id} is {len(data)} bytes, pinned {size}")
            got = _sha256_bytes(data)
            if got != digest:
                raise ValueError(f"{rid}: photo {photo_id} sha256 {got} != pinned {digest}")
            local.write_bytes(data)
        out[rid] = data
    return out


def degrade(hr: Image.Image, *, quality: int = JPEG_QUALITY) -> Image.Image:
    """The low-resolution input for a high-resolution image: bicubic 2x downsampling, then JPEG at `quality`
    (decoded back to RGB). Deterministic."""
    if not isinstance(hr, Image.Image):
        raise TypeError("hr must be a PIL image")
    if hr.width % UPSCALE or hr.height % UPSCALE:
        raise ValueError(f"hr sides must be multiples of {UPSCALE}: {hr.size}")
    if not 1 <= int(quality) <= 95:
        raise ValueError("quality must be in 1..95")
    small = hr.convert("RGB").resize((hr.width // UPSCALE, hr.height // UPSCALE), Image.BICUBIC)
    buffer = io.BytesIO()
    small.save(buffer, format="JPEG", quality=int(quality))
    return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")


def centre_crop(image: Image.Image, size: int = HR_CROP) -> Image.Image:
    """The centred `size` x `size` crop of an RGB image (raises when the image is smaller)."""
    if image.width < size or image.height < size:
        raise ValueError(f"image {image.size} is smaller than the {size} px crop")
    left, top = (image.width - size) // 2, (image.height - size) // 2
    return image.convert("RGB").crop((left, top, left + size, top + size))


def read_corpus(
    files: Mapping[str, bytes],
    *,
    crop: int = HR_CROP,
    quality: int = JPEG_QUALITY,
    records: Sequence[tuple] | None = None,
) -> list[dict[str, Any]]:
    """Decode the verified photo bytes into `{id, lr, hr}` records (HR = centred crop, LR = degrade(HR)) with the
    species as `category` and the observation provenance. `records` defaults to the 360-photo sample."""
    out = []
    for rid, label, photo_id, obs_id, user, _size, _digest, _ext in SAMPLE_RECORDS if records is None else records:
        if rid not in files:
            raise ValueError(f"corpus is missing {rid}")
        image = Image.open(io.BytesIO(files[rid]))
        image.load()
        if max(image.size) > MAX_IMAGE_SIDE:
            raise ValueError(f"{rid}: served photo {image.size} exceeds MAX_IMAGE_SIDE={MAX_IMAGE_SIDE}")
        hr = centre_crop(image, crop)
        out.append(
            {
                "id": rid,
                "hr": hr,
                "lr": degrade(hr, quality=quality),
                "category": label,
                "species": SPECIES[label][1],
                "inat_photo_id": photo_id,
                "inat_observation_url": observation_url(obs_id),
                "observer": user,
            }
        )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded stratified draw per species: `sizes` records per category for train / validation / test."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_label.setdefault(str(record.get("category", "")), []).append(dict(record))
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in sizes}
    for label in sorted(by_label):
        pool = by_label[label]
        rng.shuffle(pool)
        needed = sum(sizes.values())
        if len(pool) < needed:
            raise ValueError(f"{label}: only {len(pool)} records available, need {needed}")
        cursor = 0
        for name, per in sizes.items():
            out[name].extend(pool[cursor : cursor + per])
            cursor += per
    for part in out.values():
        rng.shuffle(part)
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir)), seed=seed)


def fetch_fresh_test(*, cache_dir: str | Path | None = None, fetcher: Any = None) -> list[dict[str, Any]]:
    """The 144 fresh-test records (digest-verified photos, the same crop and degradation as the sample)."""
    return read_corpus(fetch_corpus(cache_dir=cache_dir, fetcher=fetcher, records=FRESH_TEST_RECORDS), records=FRESH_TEST_RECORDS)


def check_fresh_disjoint(fresh: Sequence[Mapping[str, Any]], pinned: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Refuse a fresh test that shares a photo, an observation, an observer or a decoded image with `pinned` (the
    train / validation / exposed development test records). Returns the counts that were checked."""
    keys = ("inat_photo_id", "inat_observation_url", "observer")
    for key in keys:
        shared = {r.get(key) for r in fresh} & {r.get(key) for r in pinned}
        shared.discard(None)
        if shared:
            raise ValueError(f"the fresh test shares {key} {sorted(map(str, shared))[:3]} with the sample")
    pixels = {image_digest(r["hr"]) for r in pinned}
    clash = [r["id"] for r in fresh if image_digest(r["hr"]) in pixels]
    if clash:
        raise ValueError(f"fresh-test images identical to sample images: {clash[:3]}")
    return {"fresh": len(fresh), "pinned": len(pinned), "observers_shared": 0, "images_shared": 0}


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _open(image: Any, where: str) -> Image.Image:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{where}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{where}: must be a PIL.Image.Image or a file path")
    return image.convert("RGB")


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/lr/hr")
    for key in ("id", "lr", "hr"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    hr = _open(record["hr"], f"{where}.hr")
    lr = _open(record["lr"], f"{where}.lr")
    if min(hr.size) < MIN_HR_SIDE or max(hr.size) > MAX_HR_SIDE:
        raise ValueError(f"{where}: hr sides must be within {MIN_HR_SIDE}..{MAX_HR_SIDE} px, got {hr.size}")
    if hr.width % (UPSCALE * WINDOW) or hr.height % (UPSCALE * WINDOW):
        raise ValueError(f"{where}: hr sides must be multiples of {UPSCALE * WINDOW} px, got {hr.size}")
    if (lr.width * UPSCALE, lr.height * UPSCALE) != hr.size:
        raise ValueError(f"{where}: lr {lr.size} is not hr {hr.size} downscaled by {UPSCALE}")
    item = {"id": rid, "lr": lr, "hr": hr}
    category = record.get("category")
    if category is not None:
        if not isinstance(category, str) or not 1 <= len(category.strip()) <= MAX_CATEGORY_CHARS:
            raise ValueError(f"{where}: category must be a str of 1..{MAX_CATEGORY_CHARS} characters")
        item["category"] = category.strip()
    for key in ("species", "observer", "inat_photo_id", "inat_observation_url"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
) -> dict[str, Any]:
    """Structural validation of a paired dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, lr, hr} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked, ids, categories = [], set(), {}
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if "category" in item:
            categories[item["category"]] = categories.get(item["category"], 0) + 1
        checked.append(item)
    sides = [max(r["hr"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "category_counts": dict(sorted(categories.items())),
        "hr_side": {"min": min(sides), "max": max(sides)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size-prefixed) — the identity a split is made disjoint on."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, hr pixel digest, lr pixel digest)."""
    parts = sorted(f"{r['id']}:{image_digest(r['hr'])}:{image_digest(r['lr'])}" for r in records)
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no high-resolution image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["hr"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def observer_overlap(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, int]:
    """How many observers contributed photos to more than one split (an observation, not an assertion)."""
    seen: dict[str, set[str]] = {}
    for name, records in splits.items():
        for record in records:
            if record.get("observer"):
                seen.setdefault(str(record["observer"]), set()).add(name)
    return {"observers": len(seen), "in_more_than_one_split": sum(1 for s in seen.values() if len(s) > 1)}


def _split_sizes(n: int, val_fraction: float, test_fraction: float) -> tuple[int, int, int]:
    """(train, validation, test) that `split_dataset` cuts from `n` distinct images."""
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    return n - n_test - n_val, n_val, n_test


def min_byod_records(*, val_fraction: float = 0.15, test_fraction: float = 0.2) -> dict[str, int]:
    """The smallest dataset `split_dataset` accepts with these fractions: `MIN_RECORDS` training images and (when
    `val_fraction` > 0) at least one validation image, after at least one test image is held out. With the default
    fractions that is 12 distinct usable photographs, split 8 / 2 / 2."""
    for n in range(1, MAX_RECORDS + 1):
        train, val, test = _split_sizes(n, val_fraction, test_fraction)
        if train >= MIN_RECORDS and (val >= 1 or val_fraction == 0):
            return {"total": n, "train": train, "validation": val, "test": test}
    raise ValueError("no dataset size satisfies these fractions")


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating high-resolution images. The
    training split must hold `MIN_RECORDS` images and the validation split (when `val_fraction` > 0) at least one; the
    refusal names the split and the minimum total (`min_byod_records`)."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = image_digest(record["hr"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_train, n_val, n_test = _split_sizes(n, val_fraction, test_fraction)
    if n_train < MIN_RECORDS or (val_fraction > 0 and n_val < 1):
        need = min_byod_records(val_fraction=val_fraction, test_fraction=test_fraction)
        short = "training split" if n_train < MIN_RECORDS else "validation split"
        raise ValueError(
            f"{n} distinct images ({len(checked) - n} duplicate image(s) removed) leave the {short} with "
            f"{n_train if n_train < MIN_RECORDS else n_val} image(s); the training split needs {MIN_RECORDS} and the "
            f"validation split 1, so supply at least {need['total']} distinct usable photographs (split {need['train']} / "
            f"{need['validation']} / {need['test']})"
        )
    return {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif")


def _skipped_member(relative: str) -> bool:
    """Operating-system litter that is never data: macOS resource forks and hidden files."""
    return any(part == "__MACOSX" or part.startswith(".") for part in Path(relative).parts)


def load_byod_dataset(
    path: str | Path, *, crop: int = HR_CROP, quality: int = JPEG_QUALITY, report: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Records from a directory or zip of high-resolution images (optionally with one `labels.csv`: `id`, `file`,
    `category`): each image is centre-cropped to `crop` px and its LR input generated with `degrade`, so BYOD pairs
    follow the same degradation as the sample. `labels.csv` rows name images by their path relative to the folder
    holding `labels.csv` (a bare file name also matches when it is unique). Two images with the same file name in
    different folders are refused, never overwritten. Nothing is dropped silently: images smaller than the crop,
    `__MACOSX` entries, hidden files and non-image files are skipped and listed with the reason in
    `report["skipped"]`."""
    source = Path(path)
    found: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                found[file.relative_to(source).as_posix()] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    found[info.filename] = archive.read(info)  # read in memory; no extractall
    else:
        raise ValueError(f"{source} is neither a directory nor a zip file")
    skipped: list[dict[str, str]] = []
    images: dict[str, bytes] = {}
    by_name: dict[str, str] = {}
    tables = []
    for rel, data in sorted(found.items()):
        name = Path(rel).name
        if _skipped_member(rel):
            skipped.append({"file": rel, "reason": "operating-system file (__MACOSX or hidden)"})
        elif name == "labels.csv":
            tables.append(rel)
        elif Path(name).suffix.lower() not in _IMAGE_SUFFIXES:
            skipped.append({"file": rel, "reason": "not an image file"})
        else:
            if name in by_name:
                raise ValueError(f"two images share the file name {name!r} ({by_name[name]} and {rel}); rename one of them")
            by_name[name] = rel
            images[rel] = data
    if len(tables) > 1:
        raise ValueError(f"BYOD data holds more than one labels.csv: {', '.join(tables)}")
    labels: dict[str, tuple[str, str]] = {}
    if tables:
        base = Path(tables[0]).parent
        reader = csv.DictReader(io.StringIO(found[tables[0]].decode("utf-8-sig")))
        if "file" not in (reader.fieldnames or []):
            raise ValueError("labels.csv must have the column file (and optionally id, category)")
        for line, row in enumerate(reader, start=2):
            listed = str(row.get("file", "") or "").strip()
            rel = (base / listed).as_posix() if str(base) != "." else listed
            if rel not in images and listed in by_name:
                rel = by_name[listed]
            if rel not in images:
                raise ValueError(f"labels.csv line {line} (file {listed!r}): names an image that is not in the dataset")
            labels[rel] = (str(row.get("id", "") or "").strip(), str(row.get("category", "") or "").strip())
    out = []
    for rel, data in images.items():
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"BYOD file is not a decodable image: {rel}") from exc
        if min(image.size) < crop:
            skipped.append({"file": rel, "reason": f"smaller than the {crop}-px crop ({image.size[0]}x{image.size[1]})"})
            continue
        hr = centre_crop(image, crop)
        rid, category = labels.get(rel, ("", ""))
        record: dict[str, Any] = {"id": rid or re.sub(r"[^A-Za-z0-9_.:-]", "_", Path(rel).stem)[:64], "hr": hr, "lr": degrade(hr, quality=quality)}
        if category:
            record["category"] = category
        out.append(record)
    if report is not None:
        report.update({"images": len(images), "used": len(out), "labelled": len(labels), "skipped": skipped})
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A summary table (id, category, hr size, lr size, provenance), not the BYOD format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "hr_width", "hr_height", "lr_width", "lr_height", "observation"])
        for r in records:
            writer.writerow([r["id"], r.get("category", ""), r["hr"].width, r["hr"].height, r["lr"].width, r["lr"].height, r.get("inat_observation_url", "")])
    return out


def export_pairs(records: Sequence[Mapping[str, Any]], directory: str | Path) -> Path:
    """Write every record's HR and LR image as PNG under `directory` (`<id>_hr.png`, `<id>_lr.png`) with an
    `index.json`; the shape a BYOD user can inspect."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for r in records:
        r["hr"].save(out / f"{r['id']}_hr.png")
        r["lr"].save(out / f"{r['id']}_lr.png")
        index.append({"id": r["id"], "category": r.get("category"), "hr": f"{r['id']}_hr.png", "lr": f"{r['id']}_lr.png"})
    (out / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------------------------
# Pre-registered selection grid and the score-once rule (review R88-M1/M3, owner rulings 2026-10-17;
# docs/swin2sr-preregistration.md was committed before any grid run)
# ---------------------------------------------------------------------------------------------------------

GRID: dict[str, dict[str, Any]] = {
    "C0": {"trainable_stages": 1, "lr": 1e-4},  # the earlier default
    "C1": {"trainable_stages": 1, "lr": 2e-4},
    "C2": {"trainable_stages": 2, "lr": 1e-4},
}
GRID_DEFAULT = "C0"
GRID_MARGIN_DB = 0.10  # practical tolerance on best-epoch validation PSNR, not a significance threshold
SELECTION_METRIC = "best-epoch validation PSNR on the 48 validation pairs"
GRID_EPOCHS = 5
GRID_BATCH_SIZE = 8
ADAPT_SEED = 0
TEST_LOOKS = ("pre-registered evaluation (this runtime)", "reproduction", "not scored (exploratory)")
# The grid choice and the evaluation of record from the FIRST hosted run under the pre-registered settings
# (docs/release-verification.md, "Hosted T4 evaluation of record (2026-10-18)"; executed notebook and log in
# docs/execution-evidence/2026-10-18-e6a7fa2/). A CPU run (grid off) uses this choice and says so; any later pass under
# the pre-registered settings is a reproduction of this evaluation and writes `_reproduction_<n>` files.
RECORDED_SELECTION: dict[str, Any] | None = {
    "chosen": "C0",
    "model": "C0",
    "best_val_psnr": {"C0": 26.784, "C1": 26.833, "C2": 26.933},
    "frozen_val_psnr": 25.899,
    "max_memory_allocated": {"C0": 6783353856, "C1": 6783229440, "C2": 12817431040},
    "reason": "C1 +0.049 dB over C0 (< 0.10); C2 +0.149 dB over C0 but < 0.10 dB over C1 on unrounded scores (0.100 at 3 decimals), so the two-stage rule fails; C0 is kept",
    "date": "2026-10-18",
    "commit": "e6a7fa2",
    "blob": "eb13bd63",
    "evidence": "docs/execution-evidence/2026-10-18-e6a7fa2/",
}
RECORDED_EVALUATION: dict[str, Any] | None = {
    "label": "pre-registered evaluation of record",
    "date": "2026-10-18",
    "executor": "Colab CLI on a fresh T4 session",
    "commit": "e6a7fa2",
    "blob": "eb13bd63",
    "evidence": "docs/execution-evidence/2026-10-18-e6a7fa2/",
    "config": "C0",
    "model": "C0",
    "fresh_test": {
        "n": 144,
        "psnr": {"nearest": 23.83, "bicubic": 24.868, "frozen": 24.508, "selected": 25.398},
        "ssim": {"nearest": 0.715, "bicubic": 0.753, "frozen": 0.744, "selected": 0.773},
        "vs_frozen": {"psnr_mean_diff": 0.89, "ci95": [0.8168, 0.967], "wins": 144, "losses": 0, "ssim_mean_diff": 0.02915, "verdict": "improved"},
        "vs_bicubic": {"psnr_mean_diff": 0.5302, "ci95": [0.4614, 0.6042], "wins": 142, "losses": 2, "ssim_mean_diff": 0.02013, "verdict": "improved"},
    },
}
# Score-once rule item 1: the fresh test is scored only when EVERY setting below holds (the grid supplies the config).
PREREGISTERED_SETTINGS: dict[str, Any] = {
    "data": "sample",
    "sample_digest": SAMPLE_DIGEST,
    "split_seed": SAMPLE_SEED,
    "split_per_species": dict(SAMPLE_SPLIT),
    "fresh_test_digest": FRESH_TEST_DIGEST,
    "fresh_test_records": len(FRESH_TEST_RECORDS),
    "hr_crop": HR_CROP,
    "jpeg_quality": JPEG_QUALITY,
    "model_revision": MODEL_REVISION,
    "epochs": GRID_EPOCHS,
    "batch_size": GRID_BATCH_SIZE,
    "adapt_seed": ADAPT_SEED,
    "grid": True,
    "grid_configs": {name: dict(config) for name, config in GRID.items()},
    "grid_margin_db": GRID_MARGIN_DB,
    "selection_metric": SELECTION_METRIC,
}


class RunStopped(RuntimeError):
    """A refused input or an invalidated run: the notebook shows only this message (no traceback) and stops."""

    dimer_stop = True

    def _render_traceback_(self):
        # IPython shows this instead of a traceback when the cell runs in the kernel itself (routing disabled).
        return [str(self)]


def preregistration_differences(settings: Mapping[str, Any]) -> list[str]:
    """Every pre-registered setting that `settings` misses or changes, as readable strings (empty: the run is under
    the pre-registered settings)."""
    out = []
    for key, expected in PREREGISTERED_SETTINGS.items():
        got = settings.get(key, "<missing>")
        same = (
            isinstance(expected, float)
            and isinstance(got, int | float)
            and not isinstance(got, bool)
            and abs(float(got) - expected) <= 1e-12 * max(1.0, abs(expected))
        )
        if not (same or got == expected):
            out.append(f"{key}: {got!r} (pre-registered {expected!r})")
    return out


def _margin_text(diff: float, margin: float, *, tol: float = 1e-9) -> str:
    """`diff` in dB with as many decimals as it takes for the printed value to fall on the same side of `margin` as the
    unrounded one (3 when that is enough): a 0.0996 dB margin never prints as 0.100."""
    passes = diff >= margin - tol
    for digits in range(3, 10):
        text = f"{diff:+.{digits}f}"
        if (float(text) >= margin - tol) == passes:
            return text
    return f"{diff:+.9f}"


def select_config(
    results: Mapping[str, Mapping[str, Any]], *, default: str = GRID_DEFAULT, margin: float = GRID_MARGIN_DB
) -> dict[str, Any]:
    """Apply the pre-registered rules to `adapt()` results keyed by grid name, on validation PSNR only.

    A config's score is the validation PSNR of the epoch `adapt` kept. A one-stage challenger replaces `default`
    only if it beats it by at least `margin` dB; a config with more trainable stages must beat `default` AND the
    better one-stage config by at least `margin`. Within tolerance the default (smaller, current) config is kept.
    The frozen model is kept unless the selected config beats epoch 0 (the frozen model) by at least `margin`."""
    if default not in results:
        raise ValueError(f"the default config {default!r} has no result")
    tol = 1e-9
    scores, table = {}, {}
    for name, result in results.items():
        history = result["history"]
        if any(entry.get("val") is None for entry in history):
            raise ValueError(f"config {name!r} has no validation curve; the grid selects on validation only")
        table[name] = [round(float(entry["val"]["psnr"]), 3) for entry in history]
        scores[name] = float(history[int(result["best_epoch"])]["val"]["psnr"])
    stages = {name: int(results[name].get("trainable_stages", GRID.get(name, {}).get("trainable_stages", 1))) for name in results}
    base_stages = stages[default]
    one = [name for name in results if stages[name] == base_stages]
    better_one = [name for name in one if name != default and scores[name] - scores[default] >= margin - tol]
    chosen = max(better_one, key=lambda name: scores[name]) if better_one else default
    best_one = max(scores[name] for name in one)
    bigger = [
        name
        for name in results
        if stages[name] > base_stages and scores[name] - scores[default] >= margin - tol and scores[name] - best_one >= margin - tol
    ]
    if bigger:
        chosen = max(bigger, key=lambda name: scores[name])
    frozen = float(results[chosen]["history"][0]["val"]["psnr"])
    keep_frozen = scores[chosen] - frozen < margin - tol
    best_one_name = max(one, key=lambda name: scores[name])
    threshold = f"{margin:.2f}"
    margins: dict[str, dict[str, float]] = {}
    clauses = []
    for name in results:
        if name == default:
            continue
        over_default = scores[name] - scores[default]
        margins[name] = {f"vs_{default}": round(over_default, 9)}
        first = f"{_margin_text(over_default, margin)} dB over {default} ({'>=' if over_default >= margin - tol else '<'} {threshold})"
        if stages[name] == base_stages:
            clauses.append(f"{name} is {first}")
            continue
        over_best = scores[name] - best_one
        margins[name][f"vs_{best_one_name}"] = round(over_best, 9)
        second = f"{_margin_text(over_best, margin)} dB over the better one-stage config {best_one_name} ({'>=' if over_best >= margin - tol else '<'} {threshold})"
        verdict = "both clauses hold" if name in bigger else "so the more-stages rule fails"
        clauses.append(f"{name} ({stages[name]} stages) is {first} and {second}, {verdict}")
    head = (
        f"{chosen} ({scores[chosen]:.3f} dB) is selected"
        if chosen != default
        else f"{default} ({scores[default]:.3f} dB) is kept, as no challenger clears its rule"
    )
    reason = head + (": " + "; ".join(clauses) if clauses else "")
    if keep_frozen:
        reason += f"; but it does not beat the frozen model ({frozen:.3f} dB) by {threshold} dB, so the frozen model is kept"
    return {
        "configs": {
            name: {**{k: results[name].get(k) for k in ("trainable_stages", "lr", "best_epoch", "n_trainable")}, "best_val_psnr": round(scores[name], 3)}
            for name in results
        },
        "validation_psnr_by_epoch": table,
        "frozen_val_psnr": round(frozen, 3),
        "default": default,
        "margin_db": margin,
        "chosen": chosen,
        "model": "frozen" if keep_frozen else chosen,
        "reason": reason,
        "margins_db": margins,
        "selection_split": "validation (48 pairs; no test crop plays a part)",
    }


def model_state_check(adapter: Mapping[str, Any] | None, model: str, config: Mapping[str, Any] | None, *, epochs: int, batch_size: int) -> list[str]:
    """Problems with the model in memory before the fresh test is scored (empty = it is the one the pre-registered
    procedure produced): the frozen model has no adapter; an adapted model carries the selected config's stages and
    learning rate, the pre-registered epochs, batch size and seed, and the full 216 / 48 pairs."""
    if model == "frozen":
        return [] if adapter is None else ["the frozen model was selected but an adapter is loaded"]
    if adapter is None or config is None:
        return ["no adapter is loaded for the selected config"]
    expected = {
        "trainable_stages": config["trainable_stages"],
        "lr": float(config["lr"]),
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": ADAPT_SEED,
        "n_train": SAMPLE_SPLIT["train"] * len(SPECIES),
        "n_val": SAMPLE_SPLIT["validation"] * len(SPECIES),
    }
    return [f"{k}: {adapter.get(k)!r} (expected {v!r})" for k, v in expected.items() if adapter.get(k) != v]


def test_look_for(preregistered: bool, *, earlier_scored: int, recorded: Mapping[str, Any] | None) -> str:
    """Score-once labels: a run under the pre-registered settings is the pre-registered evaluation of this runtime
    unless an evaluation of record exists or this runtime already scored one (then it is a reproduction)."""
    if not preregistered:
        return TEST_LOOKS[2]
    return TEST_LOOKS[1] if earlier_scored or recorded else TEST_LOOKS[0]


test_look_for.__test__ = False  # a library helper, not a pytest test


def output_suffix(test_look: str, run_history: Sequence[Mapping[str, Any]]) -> str:
    """'' for the pre-registered evaluation; `_reproduction_<n>` / `_exploratory_<n>` otherwise (a reproduction of
    RECORDED_EVALUATION included), so neither ever overwrites the pre-registered exports."""
    if test_look == TEST_LOOKS[2]:
        return f"_exploratory_{1 + sum(h['test_look'] == TEST_LOOKS[2] for h in run_history)}"
    if test_look == TEST_LOOKS[1]:
        # numbered from 1 whether the reproduction follows this runtime's own first pass or RECORDED_EVALUATION
        return f"_reproduction_{1 + sum(h['test_look'] == TEST_LOOKS[1] for h in run_history)}"
    return ""
