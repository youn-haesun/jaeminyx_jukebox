import jukebox
import torch as t
import librosa
import os
from jukebox.make_models import make_vqvae, make_prior, MODELS, make_model
from jukebox.hparams import Hyperparams, setup_hparams
from jukebox.sample import sample_single_window, _sample, \
                           sample_partial_window, upsample
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.torch_utils import empty_cache
rank, local_rank, device = setup_dist_from_mpi()

model = "1b_lyrics"     
hps = Hyperparams()
hps.sr = 44100
hps.n_samples = 3 if model=='5b_lyrics' else 1
hps.name = 'samples'
chunk_size = 16 if model=="5b_lyrics" else 16
max_batch_size = 3 if model=="5b_lyrics" else 1
hps.levels = 3
hps.hop_fraction = [.5,.5,.125]

vqvae, *priors = MODELS[model]
vqvae = make_vqvae(setup_hparams(vqvae, dict(sample_length = 1048576)), device)
top_prior = make_prior(setup_hparams(priors[-1], dict()), vqvae, device)

sample_length_in_seconds = 60

hps.sample_length = (int(sample_length_in_seconds*hps.sr)//top_prior.raw_to_tokens)*top_prior.raw_to_tokens
assert hps.sample_length >= top_prior.n_ctx*top_prior.raw_to_tokens, f'Please choose a larger sampling rate'

metas = [dict(artist = "Eminem",
            genre = "Hip Hop",
            total_length = hps.sample_length,
            offset = 0,
            lyrics = """Look, if you had one shot, or one opportunity
                    To seize everything you ever wanted, in one moment
                    Would you capture it, or just let it slip?
                    Yo! His palms are sweaty, knees weak, arms are heavy
                    There's vomit on his sweater already: Mom's spaghetti
                    He's nervous, but on the surface he looks calm and ready
                    To drop bombs, but he keeps on forgetting
                    What he wrote down, the whole crowd goes so loud
                    He opens his mouth, but the words won't come out
                    He's choking, how? Everybody's joking now
                    The clock's run out, time's up, over, blaow!
                    Snap back to reality, ope there goes gravity, ope
                    There goes Rabbit, he choked, he's so mad but he won't
                    Give up that easy, no, he won't have it, he knows
                    His whole back's to these ropes, it don't matter, he's dope
                    He knows that but he's broke, he's so stagnant, he knows
                    When he goes back to this mobile home, that's when it's
                    Back to the lab again yo, this whole rhapsody
                    Better go capture this moment and hope it don't pass him, and
                    You better lose yourself in the music
                    The moment, you own it, you better never let it go
                    You only get one shot, do not miss your chance to blow
                    This opportunity comes once in a lifetime, yo
                    You better lose yourself in the music
                    The moment, you own it, you better never let it go
                    You only get one shot, do not miss your chance to blow
                    This opportunity comes once in a lifetime, yo
                    You better...
                    His soul's escaping through this hole that is gaping
                    This world is mine for the taking, make me king
                    As we move toward a New World Order
                    A normal life is boring; but superstardom's
                    Close to post-mortem, it only grows harder
                    Homie grows hotter, he blows, it's all over
                    These hoes is all on him, coast-to-coast shows
                    He's known as the Globetrotter, lonely roads
                    God only knows, he's grown farther from home, he's no father
                    He goes home and barely knows his own daughter
                    But hold your nose, 'cause here goes the cold water
                    These hoes don't want him no mo', he's cold product
                    They moved on to the next schmoe who flows
                    He nose-dove and sold nada, and so the soap opera
                    Is told, it unfolds, I suppose it's old, partner
                    But the beat goes on: da da dum da dum da da da da
                    You better lose yourself in the music
                    The moment, you own it, you better never let it go
                    You only get one shot, do not miss your chance to blow
                    This opportunity comes once in a lifetime, yo
                    You better lose yourself in the music
                    The moment, you own it, you better never let it go
                    You only get one shot, do not miss your chance to blow
                    This opportunity comes once in a lifetime, yo
                    You better...
                    No more games, I'ma change what you call rage
                    Tear this motherfuckin' roof off like two dogs caged
                    I was playin' in the beginning, the mood all changed
                    I've been chewed up and spit out and booed off stage
                    But I kept rhymin' and stepped right in the next cypher
                    Best believe somebody's payin' the Pied Piper
                    All the pain inside amplified by the
                    Fact that I can't get by with my 9-to-5
                    And I can't provide the right type of life for my family
                    'Cause man, these goddamn food stamps don't buy diapers
                    And there's no movie, there's no Mekhi Phifer, this is my life
                    And these times are so hard, and it's gettin' even harder
                    Tryna feed and water my seed, plus teeter-totter
                    Caught up between bein' a father and a prima donna
                    Baby mama drama, screamin' on her, too much for me to wanna
                    Stay in one spot, another day of monotony's
                    Gotten me to the point I'm like a snail, I've got
                    To formulate a plot or end up in jail or shot
                    Success is my only motherfuckin' option, failure's not
                    Mom, I love you, but this trailer's got
                    To go; I cannot grow old in Salem's Lot
                    So here I go, it's my shot: feet, fail me not
                    This may be the only opportunity that I got
                    You better lose yourself in the music
                    The moment, you own it, you better never let it go (go)
                    You only get one shot, do not miss your chance to blow
                    This opportunity comes once in a lifetime, yo
                    You better lose yourself in the music
                    The moment, you own it, you better never let it go (go)
                    You only get one shot, do not miss your chance to blow
                    This opportunity comes once in a lifetime, yo
                    You better...
                    You can do anything you set your mind to, man
            """,
            ),
          ] * hps.n_samples
labels = [None, None, top_prior.labeller.get_batch_labels(metas, 'cuda')]

sampling_temperature = .98

lower_batch_size = 16
max_batch_size = 3 if model == "5b_lyrics" else 16
lower_level_chunk_size = 32
chunk_size = 16 if model == "5b_lyrics" else 32
sampling_kwargs = [dict(temp=.99, fp16=True, max_batch_size=lower_batch_size,
                        chunk_size=lower_level_chunk_size),
                    dict(temp=0.99, fp16=True, max_batch_size=lower_batch_size,
                         chunk_size=lower_level_chunk_size),
                    dict(temp=sampling_temperature, fp16=True, 
                         max_batch_size=max_batch_size, chunk_size=chunk_size)]

zs = [t.zeros(hps.n_samples,0,dtype=t.long, device='cuda') for _ in range(len(priors))]
zs = _sample(zs, labels, sampling_kwargs, [None, None, top_prior], [2], hps)

if True:
  del top_prior
  empty_cache()
  top_prior=None
upsamplers = [make_prior(setup_hparams(prior, dict()), vqvae, 'cpu') for prior in priors[:-1]]
labels[:2] = [prior.labeller.get_batch_labels(metas, 'cuda') for prior in upsamplers]

zs = upsample(zs, labels, sampling_kwargs, [*upsamplers, top_prior], hps)

del upsamplers
empty_cache()
