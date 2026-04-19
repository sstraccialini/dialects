# this script downloads and extracts the
# nap, vec, pms, lmo, scn, eml, lld, lij, sc, fur, roa_tara
# wikipedias from 01.03.2022

wget https://dumps.wikimedia.org/other/mediawiki_content_current/napwiki/2026-04-01/xml/bzip2/napwiki-2026-04-01-p1p66122.xml.bz2
wget https://dumps.wikimedia.org/other/mediawiki_content_current/vecwiki/2026-04-01/xml/bzip2/vecwiki-2026-04-01-p3p154496.xml.bz2
# wget https://dumps.wikimedia.org/pmswiki/20220301/pmswiki-20220301-pages-articles-multistream.xml.bz2
 wget https://dumps.wikimedia.org/other/mediawiki_content_current/lmowiki/2026-04-01/xml/bzip2/lmowiki-2026-04-01-p1p279182.xml.bz2
 wget https://dumps.wikimedia.org/other/mediawiki_content_current/scnwiki/2026-04-01/xml/bzip2/scnwiki-2026-04-01-p1p67086.xml.bz2
# wget https://dumps.wikimedia.org/emlwiki/20220301/emlwiki-20220301-pages-articles-multistream.xml.bz2
# wget https://dumps.wikimedia.org/lldwiki/20220301/lldwiki-20220301-pages-articles-multistream.xml.bz2
# wget https://dumps.wikimedia.org/lijwiki/20220301/lijwiki-20220301-pages-articles-multistream.xml.bz2
# wget https://dumps.wikimedia.org/scwiki/20220301/scwiki-20220301-pages-articles-multistream.xml.bz2
# wget https://dumps.wikimedia.org/furwiki/20220301/furwiki-20220301-pages-articles-multistream.xml.bz2
#wget https://dumps.wikimedia.org/roa_tarawiki/20220301/roa_tarawiki-20220301-pages-articles-multistream.xml.bz2


python -m wikiextractor.WikiExtractor napwiki-2026-04-01-p1p66122.xml.bz2 -o nap_texts --json
 python -m wikiextractor.WikiExtractor vecwiki-2026-04-01-p3p154496.xml.bz2 -o vec_texts --json
# python -m wikiextractor.WikiExtractor pmswiki-20220301-pages-articles-multistream.xml.bz2 -o pms_texts --json
 python -m wikiextractor.WikiExtractor lmowiki-2026-04-01-p1p279182.xml.bz2 -o lmo_texts --json
 python -m wikiextractor.WikiExtractor scnwiki-2026-04-01-p1p67086.xml.bz2 -o scn_texts --json
# python -m wikiextractor.WikiExtractor emlwiki-20220301-pages-articles-multistream.xml.bz2 -o eml_texts --json
# python -m wikiextractor.WikiExtractor lldwiki-20220301-pages-articles-multistream.xml.bz2 -o lld_texts --json
# python -m wikiextractor.WikiExtractor lijwiki-20220301-pages-articles-multistream.xml.bz2 -o lij_texts --json
# python -m wikiextractor.WikiExtractor scwiki-20220301-pages-articles-multistream.xml.bz2 -o sc_texts --json
# python -m wikiextractor.WikiExtractor furwiki-20220301-pages-articles-multistream.xml.bz2 -o fur_texts --json
#python -m wikiextractor.WikiExtractor roa_tarawiki-20220301-pages-articles-multistream.xml.bz2 -o roa_tara_texts --json

python generation.py

rm -r *_texts
rm *.xml.bz2
