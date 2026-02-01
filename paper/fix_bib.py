
import re
import os
import traceback

try:
    file_path = 'egbib.bib'
    
    print(f"Reading from {os.path.abspath(file_path)}")

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Map of Journal Short Name -> Full Booktitle
    conf_map = {
        r'\{CVPR\}': r'{Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}',
        r'\{ICCV\}': r'{Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)}',
        r'\{ECCV\}': r'{Proceedings of the European Conference on Computer Vision (ECCV)}',
        r'\{NeurIPS\}': r'{Advances in Neural Information Processing Systems (NeurIPS)}',
        r'\{ICLR\}': r'{Proceedings of the International Conference on Learning Representations (ICLR)}',
        r'\{ICML\}': r'{Proceedings of the International Conference on Machine Learning (ICML)}',
        r'\{ACL\}': r'{Proceedings of the Association for Computational Linguistics (ACL)}',
        r'\{NAACL\}': r'{Proceedings of the North American Chapter of the Association for Computational Linguistics (NAACL)}',
        r'\{EMNLP\}': r'{Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)}',
        r'\{ACM MM\}': r'{Proceedings of the ACM International Conference on Multimedia (ACM MM)}',
        r'\{ACMMM\}': r'{Proceedings of the ACM International Conference on Multimedia (ACM MM)}',
        r'\{ACM\s+MM\}': r'{Proceedings of the ACM International Conference on Multimedia (ACM MM)}',
        r'\{3DV\}': r'{Proceedings of the International Conference on 3D Vision (3DV)}'
    }

    print("Processing entries...")
    entries = re.split(r'(^@)', content, flags=re.MULTILINE)
    
    output_parts = [entries[0]]
    change_count = 0
    
    for i in range(1, len(entries), 2):
        marker = entries[i]
        body = entries[i+1]
        full_entry = marker + body
        
        is_article = full_entry.lower().startswith('@article')
        modified_entry = full_entry
        
        for key, value in conf_map.items():
            pattern = r'journal\s*=\s*' + key
            if re.search(pattern, modified_entry, re.IGNORECASE):
                if is_article:
                    modified_entry = re.sub(r'@article', r'@inproceedings', modified_entry, count=1, flags=re.IGNORECASE)
                    is_article = False
                
                # Use string replacement for the value to be safe with escaping
                # But re.sub is good. value has {} which are safe in replacement string usually
                modified_entry = re.sub(pattern, f'booktitle={value}', modified_entry, flags=re.IGNORECASE)
                change_count += 1
        
        output_parts.append(modified_entry)

    final_content = "".join(output_parts)
    
    print(f"Made {change_count} changes.")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)
        
    print("Done.")

except Exception:
    traceback.print_exc()
