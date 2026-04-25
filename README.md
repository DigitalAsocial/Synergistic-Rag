# Synergistic-Rag

A concise report evaluating a synergistic RAG architecture designed to improve contextual and semantic relevance in retrieving and analyzing scientific and ML research texts. It includes a step‑by‑step implementation, an assessment of local open‑source LLMs, and a comparison of their performance within the proposed framework.

## چکیده (Abstract in Persian)
در این گزارش، عملکرد رویکرد هم‌افزایانه در معماری‌های RAG با هدف افزایش ارتباط زمینه‌ای و معنایی پاسخ‌های مدل‌های زبانی بزرگ در بازیابی و تحلیل متون علمی و ادبیات پژوهشی حوزه علوم داده و یادگیری ماشین بررسی می‌شود. سپس نحوه‌ی پیاده‌سازی معماری ارائه‌شده در مقاله‌ی اصلی، با تمرکز بر ارزیابی خروجی مدل‌های زبانی محلی (متن‌باز)، به‌صورت مرحله‌به‌مرحله مورد مطالعه قرار می‌گیرد. در ادامه، روش ارزیابی این مدل‌ها تشریح شده و در پایان، نتایج حاصل از عملکرد معماری هم‌افزایانه در مدل‌های محلی با یکدیگر و همچنین با مدل معرفی‌شده در مقاله مقایسه می‌گردد.

**واژگان کلیدی:** مدل‌های زبانی بزرگ، بازیابی اطلاعات، پیش‌پردازش، هم‌افزایی، همامتن.

## Datasets
- [ds-tb-5-raw](https://huggingface.co/datasets/DigitalAsocial/ds-tb-5-raw)
- [ds-tb-17-raw](https://huggingface.co/datasets/DigitalAsocial/ds-tb-17-raw)
- [ds-tb-5-g](https://huggingface.co/datasets/DigitalAsocial/ds-tb-5-g)
- [ds-tb-17-g](https://huggingface.co/datasets/DigitalAsocial/ds-tb-17-g)
- [ds-tb-17-g-sns](https://huggingface.co/datasets/DigitalAsocial/ds-tb-17-g-sns)
- [ds-tb-17-g-sns-aml](https://huggingface.co/datasets/DigitalAsocial/ds-tb-17-g-sns-aml/)
- [ds-tb-17-g-sns-bge](https://huggingface.co/datasets/DigitalAsocial/ds-tb-17-g-sns-bge/)

## Embedding Models 
- [all-MiniLM-L6-v2-ds-rag](https://huggingface.co/DigitalAsocial/all-MiniLM-L6-v2-ds-rag)
- [all-mpnet-base-v2-ds-rag-5r](https://huggingface.co/DigitalAsocial/all-mpnet-base-v2-ds-rag-5r)
- [all-mpnet-base-v2-ds-rag-17r](https://huggingface.co/DigitalAsocial/all-mpnet-base-v2-ds-rag-17r)
- [all-mpnet-base-v2-ds-rag-5g](https://huggingface.co/DigitalAsocial/all-mpnet-base-v2-ds-rag-5g)
- [all-mpnet-base-v2-ds-rag-17g](https://huggingface.co/DigitalAsocial/all-mpnet-base-v2-ds-rag-17g)
- [all-mpnet-base-v2-ds-rag-17s](https://huggingface.co/DigitalAsocial/all-mpnet-base-v2-ds-rag-17s)
- [all-MiniLM-L6-v2-ds-rag-s](https://huggingface.co/DigitalAsocial/all-MiniLM-L6-v2-ds-rag-s/)
- [bge-base-en-v1.5-ds-rag-s](https://huggingface.co/DigitalAsocial/bge-base-en-v1.5-ds-rag-s/)
- [all-MiniLM-L6-v2-ds-rag (alt)](https://huggingface.co/DigitalAsocial/all-MiniLM-L6-v2-ds-rag/)
