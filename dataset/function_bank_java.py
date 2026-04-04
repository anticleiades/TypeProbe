# ---------------- signature lists (kept separate) ----------------

# ================= int -> int =================

INT_INT_FUNCS = [
    lambda x: f'return {x} + 1;',
    lambda x: f'return {x} - 1;',
    lambda x: f'int out = 1;\n\tfor (int i = 1; i <= {x}; i++) {{\n\t\tout *= i;\n\t}}\n\treturn out;',
    lambda x: f'int count = 0;\n\tint n = {x};\n\twhile (n != 0) {{\n\t\tcount += (n & 1);\n\t\tn >>= 1;\n\t}}\n\treturn count;',
    lambda x: f'int a = Math.abs({x}), b = 42;\n\twhile (b != 0) {{\n\t\tint t = b;\n\t\tb = a % b;\n\t\ta = t;\n\t}}\n\treturn a;',
    lambda x: f'return {x} * ({x} + 1) / 2;',
    lambda x: f'return {x} % 2 == 0 ? 1 : 0;',
    lambda x: f'int sign = {x} < 0 ? -1 : 1;\n\tString s = Integer.toString(Math.abs({x}));\n\tString r = new StringBuilder(s).reverse().toString();\n\treturn sign * Integer.parseInt(r);',
    lambda x: f'int bits = 0;\n\tint n = {x};\n\twhile (n > 0) {{\n\t\tbits++;\n\t\tn >>= 1;\n\t}}\n\treturn bits;',
    lambda x: f'return {x} % 7;',
];

# ================= int -> String =================

INT_STR_FUNCS = [
    lambda x: f'if ({x} < 0) return "-" + Integer.toBinaryString(-{x});\n\treturn Integer.toBinaryString({x});',
    lambda x: f'if ({x} < 0) return "-" + Integer.toHexString(-{x});\n\treturn Integer.toHexString({x});',
    lambda x: f'return {x} % 2 == 0 ? "even" : "odd";',
    lambda x: f'if ({x} > 0) return "positive";\n\tif ({x} < 0) return "negative";\n\treturn "zero";',
    lambda x: f'int[] vals = {{1000,500,100,50,10,5,1}};\n\tString[] syms = {{"M","D","C","L","X","V","I"}};\n\tStringBuilder out = new StringBuilder();\n\tfor (int i = 0; i < vals.length; i++) {{\n\t\twhile ({x} >= vals[i]) {{\n\t\t\tout.append(syms[i]);\n\t\t\t{x} -= vals[i];\n\t\t}}\n\t}}\n\treturn out.toString();',
    lambda x: f'String suffix = "th";\n\tint mod100 = {x} % 100;\n\tif (mod100 < 11 || mod100 > 13) {{\n\t\tif ({x} % 10 == 1) suffix = "st";\n\t\telse if ({x} % 10 == 2) suffix = "nd";\n\t\telse if ({x} % 10 == 3) suffix = "rd";\n\t}}\n\treturn {x} + suffix;',
    lambda x: f'return Integer.toString(Integer.toString(Math.abs({x})).length());',
    lambda x: f'String b = Integer.toBinaryString({x} & 0xff);\n\treturn "0".repeat(8 - b.length()) + b;',
    lambda x: f'if ({x} < 2) return "not prime";\n\tfor (int i = 2; i * i <= {x}; i++) {{\n\t\tif ({x} % i == 0) return "composite";\n\t}}\n\treturn "prime";',
    lambda x: f'int h = ({x} / 60) % 24;\n\tint m = {x} % 60;\n\treturn String.format("%02d:%02d", h, m);',
];

# ================= String -> String =================

STR_STR_FUNCS = [
    lambda x: f'return {x}.trim();',
    lambda x: f'return {x}.toLowerCase();',
    lambda x: f'return {x}.toUpperCase();',
    lambda x: f'return new StringBuilder({x}).reverse().toString();',
    lambda x: f'return {x}.replace(" ", "_");',
    lambda x: f'String t = {x}.trim();\n\tif (t.isEmpty()) return "";\n\tStringBuilder out = new StringBuilder();\n\tfor (String w : t.split("\\\\s+")) {{\n\t\tif (!w.isEmpty()) {{\n\t\t\tout.append(Character.toUpperCase(w.charAt(0)))\n\t\t\t   .append(w.substring(1).toLowerCase());\n\t\t}}\n\t}}\n\treturn out.toString();',
    lambda x: f'return {x} + {x};',
    lambda x: f'return {x}.substring(0, {x}.length() / 2);',
    lambda x: f'if ({x}.isEmpty()) return {x};\n\tStringBuilder out = new StringBuilder();\n\tboolean prevIsLetter = false;\n\tfor (char c : {x}.toCharArray()) {{\n\t\tif (Character.isLetter(c)) {{\n\t\t\tout.append(prevIsLetter ? Character.toLowerCase(c) : Character.toUpperCase(c));\n\t\t\tprevIsLetter = true;\n\t\t}} else {{\n\t\t\tout.append(c);\n\t\t\tprevIsLetter = false;\n\t\t}}\n\t}}\n\treturn out.toString();',
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (char c : {x}.toCharArray()) {{\n\t\tif (Character.isLetter(c)) out.append(c);\n\t}}\n\treturn out.toString();',
];

# ================= String -> int =================

STR_INT_FUNCS = [
    lambda x: f'return {x}.length();',
    lambda x: f"int c = 0;\n\tfor (char ch : {x}.toCharArray()) if (ch == 'a') c++;\n\treturn c;",
    lambda x: f'int c = 0;\n\tfor (char ch : {x}.toCharArray()) if (Character.isUpperCase(ch)) c++;\n\treturn c;',
    lambda x: f'int c = 0;\n\tfor (char ch : {x}.toCharArray()) if (Character.isDigit(ch)) c++;\n\treturn c;',
    lambda x: f"return {x}.indexOf('e');",
    lambda x: f'int c = 0;\n\tfor (char ch : {x}.toCharArray()) if ("aeiouAEIOU".indexOf(ch) >= 0) c++;\n\treturn c;',
    lambda x: f'String t = {x}.trim();\n\treturn t.isEmpty() ? 0 : t.split("\\\\s+").length;',
    lambda x: f'int sum = 0;\n\tfor (char ch : {x}.toCharArray()) sum += ch;\n\treturn sum;',
    lambda x: f"int c = 0;\n\tfor (char ch : {x}.toCharArray()) if (ch == '\\n') c++;\n\treturn c;",
    lambda x: f'if ({x}.isEmpty() || !Character.isJavaIdentifierStart({x}.charAt(0))) return 0;\n\tfor (char ch : {x}.toCharArray()) if (!Character.isJavaIdentifierPart(ch)) return 0;\n\treturn 1;',
]

# ================= int -> boolean =================

INT_BOOL_FUNCS = [
    lambda x: f'return {x} % 2 == 0;',
    lambda x: f'return {x} > 0;',
]

# ================= int -> list[int] =================

INT_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tout.add({x});\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tout.add({x});\n\tout.add({x});\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int i = 0; i < {x}; i++) out.add(i);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int i = 0; i < {x}; i++) out.add(i * i);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tout.add({x} - 1);\n\tout.add({x});\n\tout.add({x} + 1);\n\treturn out;',
]

# ================= int -> list[str] =================

INT_LIST_STR_FUNCS = [
    lambda x: f'List<String> out = new ArrayList<>();\n\tout.add(Integer.toString({x}));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tif ({x} < 0) {{\n\t\tout.add("-" + Integer.toBinaryString(-{x}));\n\t\tout.add("-" + Integer.toHexString(-{x}));\n\t}} else {{\n\t\tout.add(Integer.toBinaryString({x}));\n\t\tout.add(Integer.toHexString({x}));\n\t}}\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tString s = Integer.toString(Math.abs({x}));\n\tfor (int i = 0; i < s.length(); i++) out.add(String.valueOf(s.charAt(i)));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tout.add({x} % 2 == 0 ? "even" : "odd");\n\tif ({x} > 0) out.add("positive");\n\telse if ({x} < 0) out.add("negative");\n\telse out.add("zero");\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int i = 0; i < Math.max(0, {x}); i++) out.add(Integer.toString(i));\n\treturn out;',
];

# ================= int -> list[bool] =================

INT_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x} % 2 == 0);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x} > 0);\n\tout.add({x} < 0);\n\tout.add({x} == 0);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (int i = 0; i < Math.max(0, {x}); i++) out.add(i % 2 == 0);\n\treturn out;',
    lambda x: f'int n = {x};\n\tList<Boolean> out = new ArrayList<>();\n\twhile (n > 0) {{\n\t\tout.add((n & 1) == 1);\n\t\tn >>= 1;\n\t}}\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (int i = 0; i < Math.max(0, {x}); i++) out.add(false);\n\treturn out;',
];


# ================= boolean -> boolean =================

BOOL_BOOL_FUNCS = [
    lambda x: f'return {x} ? false : true;',
    lambda x: f'return {x} ? true : false;',
    lambda x: f'return {x};',
    lambda x: f'return !{x};',
]

# ================= String -> boolean =================

STR_BOOL_FUNCS = [
    lambda x: f'return {x}.length() == 0;',
    lambda x: f'if ({x}.isEmpty()) return false;\n\tfor (char c : {x}.toCharArray()) if (!Character.isLetter(c)) return false;\n\treturn true;',
];

# ================= boolean -> int =================

BOOL_INT_FUNCS = [
    lambda x: f'return {x} ? 1 : 0;',
    lambda x: f'return {x} ? 0 : 1;',
];

# ================= bool -> String =================

BOOL_STR_FUNCS = [
    lambda x: f'return {x} ? "" : "not_empty";',
    lambda x: f'return {x} ? "not_empty" : "";',

    lambda x: f'return {x} ? "false" : "true";',
    lambda x: f'return {x} ? "true" : "false";',

    lambda x: f'return {x} ? "0" : "1";',
    lambda x: f'return {x} ? "1" : "0";',
];


# ---------------- boolean -> list[int] ----------------

BOOL_LIST_INT_FUNCS = [
    lambda x: f'if ({x}) {{\n\t\tList<Integer> out = new ArrayList<>();\n\t\tout.add(1);\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',
    lambda x: f'if (!{x}) {{\n\t\tList<Integer> out = new ArrayList<>();\n\t\tout.add(1);\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',
];

# ---------------- boolean -> list[str] ----------------

BOOL_LIST_STR_FUNCS = [
    lambda x: f'if ({x}) {{\n\t\tList<String> out = new ArrayList<>();\n\t\tout.add("");\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',
    lambda x: f'if (!{x}) {{\n\t\tList<String> out = new ArrayList<>();\n\t\tout.add("not_empty");\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',

    lambda x: f'if ({x}) {{\n\t\tList<String> out = new ArrayList<>();\n\t\tout.add("false");\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',
    lambda x: f'if ({x}) {{\n\t\tList<String> out = new ArrayList<>();\n\t\tout.add("true");\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',

    lambda x: f'if ({x}) {{\n\t\tList<String> out = new ArrayList<>();\n\t\tout.add("0");\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',
    lambda x: f'if ({x}) {{\n\t\tList<String> out = new ArrayList<>();\n\t\tout.add("1");\n\t\treturn out;\n\t}}\n\treturn new ArrayList<>();',
];

# ---------------- list[int] -> int ----------------

LIST_INT_INT_FUNCS = [
    lambda x: f'int sum = 0;\n\tfor (int v : {x}) sum += v;\n\treturn sum;',
    lambda x: f'int out = 0;\n\tfor (int v : {x}) out += v * v;\n\treturn out;',
    lambda x: f'if ({x}.isEmpty()) return 0;\n\tint m = {x}.get(0);\n\tfor (int v : {x}) if (v > m) m = v;\n\treturn m;',
    lambda x: f'if ({x}.isEmpty()) return 0;\n\tint m = {x}.get(0);\n\tfor (int v : {x}) if (v < m) m = v;\n\treturn m;',
    lambda x: f'return {x}.size();',
];

# ---------------- list[int] -> list[int] ----------------

LIST_INT_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(v + 1);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(v * 2);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int v : {x}) if (v % 2 == 0) out.add(v);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int i = {x}.size() - 1; i >= 0; i--) out.add({x}.get(i));\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>({x});\n\tCollections.sort(out);\n\treturn out;',
];

# ---------------- list[int] -> bool ----------------

LIST_INT_BOOL_FUNCS = [
    lambda x: f'return {x}.isEmpty();',
    lambda x: f'for (int v : {x}) if (v < 0) return false;\n\treturn true;',
];

# ---------------- list[str] -> int ----------------

LIST_STR_INT_FUNCS = [
    lambda x: f'return {x}.size();',
    lambda x: f'int sum = 0;\n\tfor (String s : {x}) sum += s.length();\n\treturn sum;',
    lambda x: f'int count = 0;\n\tfor (String s : {x}) if (!s.isEmpty()) count++;\n\treturn count;',
];


# ---------------- list[str] -> str ----------------

LIST_STR_STR_FUNCS = [
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (String s : {x}) out.append(s);\n\treturn out.toString();',
    lambda x: f'return String.join(" ", {x});',
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (String s : {x}) if (!s.isEmpty()) out.append(s.charAt(0));\n\treturn out.toString();',
];


# ================= list[str] -> list[int] =================

LIST_STR_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (String s : {x}) out.add(s.length());\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (String s : {x}) {{\n\t\tint c = 0;\n\t\tfor (char ch : s.toCharArray()) if (Character.isDigit(ch)) c++;\n\t\tout.add(c);\n\t}}\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (String s : {x}) {{\n\t\tint c = 0;\n\t\tfor (char ch : s.toCharArray()) if (Character.isUpperCase(ch)) c++;\n\t\tout.add(c);\n\t}}\n\treturn out;',
    lambda x: f"List<Integer> out = new ArrayList<>();\n\tfor (String s : {x}) {{\n\t\tint c = 0;\n\t\tfor (char ch : s.toCharArray()) if (ch == 'a') c++;\n\t\tout.add(c);\n\t}}\n\treturn out;",
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (String s : {x}) {{\n\t\tString t = s.trim();\n\t\tint c = t.isEmpty() ? 0 : t.split("\\\\s+").length;\n\t\tout.add(c);\n\t}}\n\treturn out;',
];

# ---------------- list[int] -> list[bool] ----------------

LIST_INT_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(v % 2 == 0);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(v > 0);\n\treturn out;',
];

# ---------------- list[str] -> list[bool] ----------------

LIST_STR_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (String s : {x}) out.add(!s.isEmpty());\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (String s : {x}) {{\n\t\tif (s.isEmpty()) continue;\n\t\tboolean ok = true;\n\t\tfor (char ch : s.toCharArray()) if (!Character.isLetter(ch)) {{ ok = false; break; }}\n\t\tout.add(ok);\n\t}}\n\treturn out;',
];

# ---------------- list[bool] -> list[int] ----------------

LIST_BOOL_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? 1 : 0);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? 0 : 1);\n\treturn out;',
];

# ---------------- list[bool] -> list[str] ----------------

LIST_BOOL_LIST_STR_FUNCS = [
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? "1" : "0");\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? "true" : "false");\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? "" : "not_empty");\n\treturn out;',
];


# ================= String -> list[int] =================

STR_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tout.add({x}.length());\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (char c : {x}.toCharArray()) out.add((int) c);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (int i = 0; i < {x}.length(); i++) {{\n\t\tif (Character.isDigit({x}.charAt(i))) out.add(i);\n\t}}\n\treturn out;',
    lambda x: f'String t = {x}.trim();\n\tList<Integer> out = new ArrayList<>();\n\tif (t.isEmpty()) return out;\n\tfor (String w : t.split("\\\\s+")) out.add(w.length());\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tString vowels = "aeiou";\n\tfor (int j = 0; j < vowels.length(); j++) {{\n\t\tchar v = vowels.charAt(j);\n\t\tint c = 0;\n\t\tfor (int i = 0; i < {x}.length(); i++) if ({x}.charAt(i) == v) c++;\n\t\tout.add(c);\n\t}}\n\treturn out;',
]

# ================= String -> list[str] =================

STR_LIST_STR_FUNCS = [
    lambda x: f'String t = {x}.trim();\n\tList<String> out = new ArrayList<>();\n\tif (t.isEmpty()) return out;\n\tfor (String w : t.split("\\\\s+")) out.add(w);\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (char c : {x}.toCharArray()) out.add(String.valueOf(c));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tif ({x}.indexOf(\',\') >= 0) {{\n\t\tfor (String p : {x}.split(",", -1)) out.add(p);\n\t}} else {{\n\t\tout.add({x});\n\t}}\n\treturn out;',
    lambda x: f'String t = {x}.trim();\n\tList<String> out = new ArrayList<>();\n\tif (t.isEmpty()) return out;\n\tfor (String w : t.split("\\\\s+")) out.add(w.toLowerCase());\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int i = 0; i <= {x}.length(); i++) out.add({x}.substring(0, i));\n\treturn out;',
]

# ================= String -> list[bool] =================

STR_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (char c : {x}.toCharArray()) out.add(Character.isLetter(c));\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (char c : {x}.toCharArray()) out.add(Character.isDigit(c));\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (char c : {x}.toCharArray()) out.add(Character.isUpperCase(c));\n\treturn out;',
    lambda x: f'String t = {x}.trim();\n\tList<Boolean> out = new ArrayList<>();\n\tif (t.isEmpty()) return out;\n\tfor (String w : t.split("\\\\s+")) out.add(w.length() > 0);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tString vowels = "aeiouAEIOU";\n\tfor (char c : {x}.toCharArray()) out.add(vowels.indexOf(c) >= 0);\n\treturn out;',
]

# ================= boolean -> list[bool] =================

BOOL_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x});\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add(!{x});\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x});\n\tout.add(!{x});\n\treturn out;',
    lambda x: f'if ({x}) return new ArrayList<>();\n\tList<Boolean> out = new ArrayList<>();\n\tout.add(false);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x} ? true : false);\n\treturn out;',
]

# ================= list[int] -> String =================

LIST_INT_STR_FUNCS = [
    lambda x: f"StringBuilder out = new StringBuilder();\n\tfor (int i = 0; i < {x}.size(); i++) {{\n\t\tif (i > 0) out.append(',');\n\t\tout.append({x}.get(i));\n\t}}\n\treturn out.toString();",
    lambda x: f'if ({x}.isEmpty()) return "";\n\tint sum = 0;\n\tfor (int v : {x}) sum += v;\n\treturn Integer.toString(sum);',
    lambda x: f'if ({x}.isEmpty()) return "";\n\tint m = {x}.get(0);\n\tfor (int v : {x}) if (v > m) m = v;\n\treturn Integer.toString(m);',
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (int v : {x}) out.append(Math.floorMod(v, 10));\n\treturn out.toString();',
    lambda x: f"StringBuilder out = new StringBuilder();\n\tfor (int i = 0; i < {x}.size(); i++) {{\n\t\tif (i > 0) out.append(' ');\n\t\tout.append({x}.get(i));\n\t}}\n\treturn out.toString();",
]

# ================= list[int] -> list[str] =================

LIST_INT_LIST_STR_FUNCS = [
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(Integer.toString(v));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(Integer.toBinaryString(v));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(Integer.toHexString(v));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(v % 2 == 0 ? "even" : "odd");\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(Integer.toString(Math.abs(v)));\n\treturn out;',
]

# ================= list[str] -> bool =================

LIST_STR_BOOL_FUNCS = [
    lambda x: f'return {x}.isEmpty();',
    lambda x: f'for (String s : {x}) if (s == null || s.isEmpty()) return false;\n\treturn true;',
    lambda x: f"for (String s : {x}) if (s != null && s.indexOf('a') >= 0) return true;\n\treturn false;",
    lambda x: f'for (String s : {x}) {{\n\t\tif (s == null) continue;\n\t\tfor (char c : s.toCharArray()) if (Character.isDigit(c)) return true;\n\t}}\n\treturn false;',
    lambda x: f'for (String s : {x}) {{\n\t\tif (s == null) continue;\n\t\tfor (char c : s.toCharArray()) if (Character.isUpperCase(c)) return true;\n\t}}\n\treturn false;',
]

# ================= list[str] -> list[str] =================

LIST_STR_LIST_STR_FUNCS = [
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (String s : {x}) out.add(s == null ? null : s.trim());\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (String s : {x}) out.add(s == null ? null : s.toLowerCase());\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (String s : {x}) out.add(s == null ? null : s.toUpperCase());\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (String s : {x}) out.add(s == null ? null : new StringBuilder(s).reverse().toString());\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>({x});\n\tCollections.sort(out);\n\treturn out;',
]

# ================= list[bool] -> int =================

LIST_BOOL_INT_FUNCS = [
    lambda x: f'int c = 0;\n\tfor (boolean v : {x}) if (v) c++;\n\treturn c;',
    lambda x: f'return {x}.size();',
    lambda x: f'if ({x}.isEmpty()) return 0;\n\tfor (boolean v : {x}) if (!v) return 0;\n\treturn 1;',
    lambda x: f'if ({x}.isEmpty()) return 0;\n\tfor (boolean v : {x}) if (v) return 1;\n\treturn 0;',
    lambda x: f'int sum = 0;\n\tfor (boolean v : {x}) sum += (v ? 1 : 0);\n\treturn sum;',
]

# ================= list[bool] -> bool =================

LIST_BOOL_BOOL_FUNCS = [
    lambda x: f'for (boolean v : {x}) if (!v) return false;\n\treturn true;',
    lambda x: f'for (boolean v : {x}) if (v) return true;\n\treturn false;',
    lambda x: f'return {x}.isEmpty();',
    lambda x: f'if ({x}.isEmpty()) return false;\n\treturn {x}.get(0);',
    lambda x: f'if ({x}.isEmpty()) return false;\n\treturn {x}.get({x}.size() - 1);',
]

# ================= list[bool] -> String =================
# (This completes the 6x6 grid; preserves semantics similar to Python-style join/encoding.)

LIST_BOOL_STR_FUNCS = [
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (int i = 0; i < {x}.size(); i++) {{\n\t\tif (i > 0) out.append(\',\');\n\t\tout.append({x}.get(i) ? "1" : "0");\n\t}}\n\treturn out.toString();',
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (int i = 0; i < {x}.size(); i++) {{\n\t\tif (i > 0) out.append(\' \');\n\t\tout.append({x}.get(i) ? "true" : "false");\n\t}}\n\treturn out.toString();',
    lambda x: f'return {x}.isEmpty() ? "" : ({x}.get(0) ? "true" : "false");',
    lambda x: f'return {x}.isEmpty() ? "0" : Integer.toString({x}.size());',
    lambda x: f'int c = 0;\n\tfor (boolean v : {x}) if (v) c++;\n\treturn Integer.toString(c);',
]

# ================= list[bool] -> list[bool] =================

LIST_BOOL_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(!v);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (int i = {x}.size() - 1; i >= 0; i--) out.add({x}.get(i));\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (boolean v : {x}) if (v) out.add(true);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (boolean v : {x}) if (!v) out.add(false);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>({x});\n\tCollections.sort(out);\n\treturn out;',
]

# ================= float -> float =================

FLOAT_FLOAT_FUNCS = [
    lambda x: f'return {x} + 0.5f;',
    lambda x: f'return {x} * 1.1f;',
    lambda x: f'return -{x};',
    lambda x: f'return Math.abs({x});',
    lambda x: f'return {x} / 2.0f;',
]

# ================= float -> int =================

FLOAT_INT_FUNCS = [
    lambda x: f'return (int){x};',
    lambda x: f'return (int)Math.round({x});',
    lambda x: f'return (int)Math.abs({x});',
]

# ================= float -> String =================

FLOAT_STR_FUNCS = [
    lambda x: f'return Float.toString({x});',
    lambda x: f'return String.format("%.2f", {x});',
]

# ================= float -> boolean =================

FLOAT_BOOL_FUNCS = [
    lambda x: f'return {x} > 0.0f;',
    lambda x: f'return {x} == 0.0f;',
]

# ================= float -> list[int] =================

FLOAT_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tout.add((int){x});\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tout.add((int){x});\n\tout.add(((int){x}) + 1);\n\treturn out;',
]

# ================= float -> list[String] =================

FLOAT_LIST_STR_FUNCS = [
    lambda x: f'List<String> out = new ArrayList<>();\n\tout.add(Float.toString({x}));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tout.add(String.format("%.1f", {x}));\n\tout.add(String.format("%.3f", {x}));\n\treturn out;',
]

# ================= float -> list[boolean] =================

FLOAT_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x} > 0.0f);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tout.add({x} == 0.0f);\n\tout.add({x} != 0.0f);\n\treturn out;',
]

# ================= float -> list[float] =================

FLOAT_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tout.add({x});\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tout.add({x});\n\tout.add({x} / 2.0f);\n\tout.add({x} * 2.0f);\n\treturn out;',
]

# ================= int -> float =================

INT_FLOAT_FUNCS = [
    lambda x: f'return (float){x};',
    lambda x: f'return {x} / 2.0f;',
]

# ================= String -> float =================

STR_FLOAT_FUNCS = [
    lambda x: f'return (float){x}.length();',
    lambda x: f'float sum = 0.0f;\n\tfor (char c : {x}.toCharArray()) sum += c;\n\treturn sum;',
]

# ================= boolean -> float =================

BOOL_FLOAT_FUNCS = [
    lambda x: f'return {x} ? 1.0f : 0.0f;',
    lambda x: f'return {x} ? 0.0f : 1.0f;',
]

# ================= list[int] -> float =================

LIST_INT_FLOAT_FUNCS = [
    lambda x: f'if ({x}.isEmpty()) return 0.0f;\n\tint sum = 0;\n\tfor (int v : {x}) sum += v;\n\treturn ((float)sum) / {x}.size();',
    lambda x: f'int sum = 0;\n\tfor (int v : {x}) sum += v;\n\treturn (float)sum;',
]

# ================= list[String] -> float =================

LIST_STR_FLOAT_FUNCS = [
    lambda x: f'if ({x}.isEmpty()) return 0.0f;\n\tint sum = 0;\n\tfor (String s : {x}) sum += s.length();\n\treturn ((float)sum) / {x}.size();',
    lambda x: f'int sum = 0;\n\tfor (String s : {x}) sum += s.length();\n\treturn (float)sum;',
]

# ================= list[boolean] -> float =================

LIST_BOOL_FLOAT_FUNCS = [
    lambda x: f'if ({x}.isEmpty()) return 0.0f;\n\tint sum = 0;\n\tfor (boolean v : {x}) if (v) sum++;\n\treturn ((float)sum) / {x}.size();',
    lambda x: f'int sum = 0;\n\tfor (boolean v : {x}) if (v) sum++;\n\treturn (float)sum;',
]

# ================= list[float] -> float =================

LIST_FLOAT_FLOAT_FUNCS = [
    lambda x: f'if ({x}.isEmpty()) return 0.0f;\n\tfloat sum = 0.0f;\n\tfor (float v : {x}) sum += v;\n\treturn sum / {x}.size();',
    lambda x: f'float sum = 0.0f;\n\tfor (float v : {x}) sum += v;\n\treturn sum;',
    lambda x: f'if ({x}.isEmpty()) return 0.0f;\n\tfloat m = {x}.get(0);\n\tfor (float v : {x}) if (v > m) m = v;\n\treturn m;',
]

# ================= int -> list[float] =================

INT_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tout.add((float){x});\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (int i = 0; i < Math.max(0, {x}); i++) out.add(i + 0.5f);\n\treturn out;',
]

# ================= String -> list[float] =================

STR_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tout.add((float){x}.length());\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (char c : {x}.toCharArray()) out.add(((float)c) / 10.0f);\n\treturn out;',
]

# ================= boolean -> list[float] =================

BOOL_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tout.add({x} ? 1.0f : 0.0f);\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tif ({x}) {{ out.add(0.0f); out.add(1.0f); }} else {{ out.add(0.0f); }}\n\treturn out;',
]

# ================= list[int] -> list[float] =================

LIST_INT_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (int v : {x}) out.add((float)v);\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (int v : {x}) out.add(v / 2.0f);\n\treturn out;',
]

# ================= list[String] -> list[float] =================

LIST_STR_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (String s : {x}) out.add((float)s.length());\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (String s : {x}) {{\n\t\tint c = 0;\n\t\tfor (char ch : s.toCharArray()) if (Character.isDigit(ch)) c++;\n\t\tout.add((float)c);\n\t}}\n\treturn out;',
]

# ================= list[boolean] -> list[float] =================

LIST_BOOL_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? 1.0f : 0.0f);\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (boolean v : {x}) out.add(v ? 0.0f : 1.0f);\n\treturn out;',
]

# ================= list[float] -> int =================

LIST_FLOAT_INT_FUNCS = [
    lambda x: f'return {x}.size();',
    lambda x: f'float sum = 0.0f;\n\tfor (float v : {x}) sum += v;\n\treturn (int)sum;',
    lambda x: f'if ({x}.isEmpty()) return 0;\n\tfloat m = {x}.get(0);\n\tfor (float v : {x}) if (v > m) m = v;\n\treturn (int)m;',
]

# ================= list[float] -> String =================

LIST_FLOAT_STR_FUNCS = [
    lambda x: f'StringBuilder out = new StringBuilder();\n\tfor (int i = 0; i < {x}.size(); i++) {{\n\t\tif (i > 0) out.append(\',\');\n\t\tout.append(String.format("%.2f", {x}.get(i)));\n\t}}\n\treturn out.toString();',
    lambda x: f'if ({x}.isEmpty()) return "";\n\tfloat sum = 0.0f;\n\tfor (float v : {x}) sum += v;\n\treturn String.format("%.2f", sum);',
]

# ================= list[float] -> boolean =================

LIST_FLOAT_BOOL_FUNCS = [
    lambda x: f'return {x}.isEmpty();',
    lambda x: f'for (float v : {x}) if (v <= 0.0f) return false;\n\treturn true;',
]

# ================= list[float] -> list[int] =================

LIST_FLOAT_LIST_INT_FUNCS = [
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (float v : {x}) out.add((int)v);\n\treturn out;',
    lambda x: f'List<Integer> out = new ArrayList<>();\n\tfor (float v : {x}) out.add((int)Math.abs(v));\n\treturn out;',
]

# ================= list[float] -> list[String] =================

LIST_FLOAT_LIST_STR_FUNCS = [
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (float v : {x}) out.add(Float.toString(v));\n\treturn out;',
    lambda x: f'List<String> out = new ArrayList<>();\n\tfor (float v : {x}) out.add(String.format("%.1f", v));\n\treturn out;',
]

# ================= list[float] -> list[boolean] =================

LIST_FLOAT_LIST_BOOL_FUNCS = [
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (float v : {x}) out.add(v > 0.0f);\n\treturn out;',
    lambda x: f'List<Boolean> out = new ArrayList<>();\n\tfor (float v : {x}) out.add(v == 0.0f);\n\treturn out;',
]

# ================= list[float] -> list[float] =================

LIST_FLOAT_LIST_FLOAT_FUNCS = [
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (float v : {x}) out.add(v * 2.0f);\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>();\n\tfor (float v : {x}) out.add(v / 2.0f);\n\treturn out;',
    lambda x: f'List<Float> out = new ArrayList<>({x});\n\tCollections.sort(out);\n\treturn out;',
]

