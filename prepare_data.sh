
scp -r kanishk@ada:/share3/kanishk/brain2word_data.tar ./

cd brain2word_data

for f in *.tar; do tar xf "$f"; done
for f in *.tar; do rm "$f"; done

cd ..

rm brain2word_data.tar