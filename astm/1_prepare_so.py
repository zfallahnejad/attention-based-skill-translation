import os
import re
import html

for target_tag, infile_name in [("java", "JavaPosts.xml"), ("php", "PhpPosts.xml")]:
    so_input_path = "./data/" + infile_name
    so_output_path = "./data/"
    if not os.path.exists(so_output_path):
        os.makedirs(so_output_path)

    pattern_1 = re.compile("[a-zA-Z\d][,|.|?|:]\s")

    Id_regex = re.compile("(?<=Id=\")(?P<Id>.*?)(?=\" )")
    Body_regex = re.compile("(?<=Body=\")(?P<Body>.*?)(?=\" )")
    print("Preprocessing input text...")
    with open(so_output_path + "so_{}.txt".format(target_tag), "w", encoding='utf8') as post_out_file:
        with open(so_input_path, encoding='utf8') as posts_file:
            for line in posts_file:
                post_id = Id_regex.search(line).group('Id')
                print(post_id)
                body = Body_regex.search(line).group('Body')
                doc = body.lower().replace('&amp;', '&').replace("&lt;", "<").replace("&gt;", ">").replace("&#xa;", " ")
                doc = html.unescape(doc)
                for tag in ['<p>', '</p>', '<b>', "<br>", "<br/>", "<br />", "<ul>", "</ul>", "<li>", "</li>", "<dt>",
                            "<ol>", "</ol>", "<hr>", "<i>", "</i>", "<b>", "</b>", "<pre>", "</pre>", "<blockquote>",
                            "</blockquote>", "<code>", "</code>", "<hr>", "<hr/>", "<em>", "</em>", "</a>", "<strong>",
                            "</strong>", "<strike>", "<t>", "<h1>", "</h1>", "<h2>", "</h2>", "<h3>", "</h3>", "<h4>",
                            "</h4>", "<h5>", "</h5>", "<h6>", "</h6>", "<string>", "<html>", "</html>", "<head>",
                            "<dl>", "</dl>", "<dd>", "<ul>", "</ul>", "<hr />", "<table>", "</table>", "<tr>", "</tr>",
                            "<td>", "</td>", "<th>", "</th>", "<frameset>", "</frameset>", "</frame>", "<option>",
                            "<noframes>", "</noframes>", "<form>", "</form>", "</select>", "<body>", "</body>",
                            "</head>", "<tt>", "</tt>", "<cite>", "</cite>", "</font>"]:
                    doc = ' {} '.format(tag).join(doc.split(tag))
                for tag in ['<a ', '<img ', '<body ', '<font ', '<p ', "<input ", '<div ', '<hr ', "<select ",
                            '<table ', '<td ', '<tr ', '<frame ', '<frameset ']:
                    doc = ' {}'.format(tag).join(doc.split(tag))
                tokens = doc.split()
                for stop_word in ['.', ',', '?', ':']:
                    if stop_word in tokens:
                        tokens.remove(stop_word)
                doc = ' '.join(tokens)
                doc = doc.replace("\n", " ").replace("\r", " ")

                search_results = []
                for m in pattern_1.finditer(doc):
                    search_results.insert(0, m.span())
                for s in search_results:
                    doc = doc[:s[0] + 1] + " " + doc[s[1]:]

                post_out_file.write(str(post_id) + ' ' + doc + "\n")
    print("End!")
