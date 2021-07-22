import ast
import asttokens
import io
import pandas as pd
import spacy
import tqdm
import token
import tokenize
import numpy as np

from spacy.matcher import Matcher

nlp = spacy.load("en_core_web_md")

AST_TYPES = {
    ast.List: 'AstList',
    ast.ListComp: 'AstListComp',
    ast.GeneratorExp: 'AstGen',
    ast.Dict: 'AstDict',
    ast.DictComp: 'AstDictComp',
    ast.Set: 'AstSet',
    ast.SetComp: 'AstSetComp',
    ast.BoolOp: 'AstBoolOp',
    ast.Bytes: 'AstChar',
    ast.Num: 'AstNum',
    ast.Str: 'AstStr',
    ast.Tuple: 'AstTuple',
    ast.Compare: 'AstCompare'}


def remove_python_comments(original, output):
    """Removes the comments from the original string of code."""
    prev_toktype = token.INDENT
    first_line = None
    last_lineno = -1
    last_col = 0
    tokgen = tokenize.generate_tokens(original.readline)
    for toktype, ttext, (slineno, scol), (elineno, ecol), ltext in tokgen:
        if slineno > last_lineno:
            last_col = 0
        if scol > last_col:
            output.write(" " * (scol - last_col))
        if toktype == token.STRING and prev_toktype == token.INDENT:
            pass
        elif toktype == tokenize.COMMENT:
            pass
        else:
            output.write(ttext)
        prev_toktype = toktype
        last_col = ecol
        last_lineno = elineno


def get_matcher():
    """Regex matching rules for the data types that happen in python the most."""
    matcher = Matcher(nlp.vocab)
    patterns = [[{"TEXT": {"REGEX": "dict"}}], [{"TEXT": {"REGEX": "map"}}]]
    matcher.add("dict", patterns)
    patterns = [[{"TEXT": {"REGEX": "list"}}], [{"TEXT": {"REGEX": "arr"}}]]
    matcher.add("list", patterns)
    patterns = [[{"TEXT": {"REGEX": "tuple"}}]]
    matcher.add("tuple", patterns)
    patterns = [[{"TEXT": {"REGEX": "count"}}], [{"TEXT": {"REGEX": "cnt"}}]]
    matcher.add("int", patterns)
    patterns = [[{"TEXT": {"REGEX": "file"}}]]
    matcher.add("file", patterns)
    patterns = [[{"TEXT": {"REGEX": "enum"}}]]
    matcher.add("enum", patterns)
    patterns = [[{"TEXT": {"REGEX": "str"}}]]
    matcher.add("string", patterns)
    patterns = [[{"TEXT": {"REGEX": "char"}}], [
        {"TEXT": {"REGEX": "unicode"}}], [{"TEXT": {"REGEX": "ascii"}}]]
    matcher.add("string", patterns)
    patterns = [[{"TEXT": {"REGEX": "path"}}], [{"TEXT": {"REGEX": "dir"}}]]
    matcher.add("path", patterns)
    patterns = [[{"TEXT": {"REGEX": "True"}}], [
        {"TEXT": {"REGEX": "False"}}], [{"TEXT": {"REGEX": "bool"}}]]
    matcher.add("bool", patterns)
    return matcher


def add_to_matcher(token, matcher):
    """Adds the rules for the new query to regex matcher."""
    if not matcher.get(token):
        patterns = [[{"TEXT": {"REGEX": token}}]]
        matcher.add(token, patterns)

    return matcher


def get_matched_labels(code_tokens, matcher):
    """Gets the regex matching labels of the code tokens."""
    tags = [[] for t in code_tokens]
    matches = [matcher(nlp(t)) for t in code_tokens]
    for j, match in enumerate(matches):
        for match_id, start, end in match:
            # Get string representation
            string_id = nlp.vocab.strings[match_id]
            tags[j].append(string_id)
    return tags


def get_matched_labels_binary(code_tokens, nlp_cache, matcher):
    """Gets the regex matching labels of the code tokens and returns a list containing 0 and 1."""
    tags = np.zeros(len(code_tokens))
    matches = [matcher(nlp_t) for nlp_t in nlp_cache]
    for j, match in enumerate(matches):
        for match_id, start, end in match:
            tags[j] = 1
    return tags


def get_static_labels(code_tokens, atok):
    """Gets the static labels of the code tokens."""
    ast_labels = [[] for t in code_tokens]
    for node in ast.walk(atok.tree):
        for ast_type in AST_TYPES:
            if isinstance(node, ast_type):
                try:
                    for token in atok.token_range(node.first_token, node.last_token):
                        ast_labels[token[5]].append(AST_TYPES[ast_type])
                except:
                    pass
    return ast_labels


def get_static_labels_binary(code_tokens, query, static_tags, tokenizer):
    """Gets the static labels of the code tokens and returns a list containing 0 and 1."""

    def find_sublist(b, a):
        idx = []
        for i in range(0, len(b) - len(a) + 1):
            if b[i:i + len(a)] == a:
                idx.extend(list(range(i, i + len(a))))
        return idx

    def get_match(tokens, code_tokens, tokenizer):
        tokens = ' '.join(tokens) if len(tokens) > 1 else tokens[0]
        tokens = tokenizer.tokenize(tokens.lower())
        sub_idx = find_sublist(code_tokens, tokens)
        return sub_idx

    static_match = np.zeros(len(code_tokens))

    static_labels = ['list', 'dict', 'generator', 'set', 'bool', 'char', 'num', 'str', 'tuple', 'compare']
    label_exists = [label in query for label in static_labels]

    if any(label_exists):
        idx = label_exists.index(True)
        label = static_labels[idx]
        for tokens, tag in static_tags:
            exists = [True for t in tag if label in t.lower()]
            if any(exists):
                code_idx = get_match(tokens, code_tokens, tokenizer)
                static_match[code_idx] = 1

    return static_match


def new_query_matches(token, code_tokens, nlp_cache, static_tags, tokenizer):
    """Having the query token, the function gets the necessary regex and static labels, and return a list
    with all the label matches."""
    matcher = Matcher(nlp.vocab)
    matcher = add_to_matcher(token, matcher)

    regex_matcher = get_matched_labels_binary(code_tokens, nlp_cache, matcher)
    static_matcher = get_static_labels_binary(code_tokens, token, static_tags, tokenizer)

    return regex_matcher, static_matcher


def process(input_file, output_file, query_token):
    """Having the input file, it finds the regex and static labels
    for every code string and saves them as a json file."""
    csn = pd.read_json(input_file, lines=True)
    output_file_no_ext = ''.join(output_file.split('.')[:-2])
    print('output file without extension: ', output_file_no_ext)
    f = open(output_file_no_ext + '.SE_list', 'w+')

    matcher_tags_col_value = [[] for c in csn['code']]
    static_tags_col_value = [[] for c in csn['code']]
    alt_token = [[] for c in csn['code']]
    csn['regex_tags'] = matcher_tags_col_value
    csn['static_tags'] = static_tags_col_value
    csn['alt_code_tokens'] = alt_token

    matcher = get_matcher()
    add_to_matcher(query_token, matcher)

    for i, (code, doc) in tqdm.tqdm(enumerate(zip(csn['code'], csn['docstring_tokens']))):
        source_io = io.StringIO()
        remove_python_comments(io.StringIO(code), source_io)
        source = source_io.getvalue()
        source_io.close()
        try:
            atok = asttokens.ASTTokens(source, parse=True)
        except SyntaxError as se:
            f.write(str(i))
            f.write('\n')
            f.flush()
        atok.mark_tokens(atok.tree)
        all_tokens = [t[1] for t in atok.tokens]
        alt_token[i] = all_tokens

        static_tags_col_value[i] = get_static_labels(all_tokens, atok)
        matcher_tags_col_value[i] = get_matched_labels(all_tokens, matcher)

    f.close()
    csn['regex_tags'] = matcher_tags_col_value
    csn['static_tags'] = static_tags_col_value
    csn['alt_code_tokens'] = alt_token
    csn.to_json(output_file, lines=True, orient='records')
