import taichi as ti
from tolvera import Tolvera, run

'''
Instructions to run:
1. Certain python libraries must be installed for the osc_sender to run. You can install them with:

    pip install numpy aubio soundfile sounddevice python-osc

2. Run the osc_beat_visualizer.py file first
3. In a separate terminal, run the osc_sender. 
4. You should now be able to hear audio and see the beat visualizer reacting to the osc messages. Enjoy!
'''


def main(**kwargs):
    tv = Tolvera(**kwargs, species=5, particles=1000)

    # Fields to store beat state
    beat_counter = ti.field(dtype=ti.i32, shape=())  
    beat_started = ti.field(dtype=ti.i32, shape=())  # acts like a boolean
    last_beat_frame = ti.field(dtype=ti.i32, shape=())
    frame_counter = ti.field(dtype=ti.i32, shape=())
    beat_started[None] = 0


    # Capture beat signal from OSC
    @tv.osc.map.receive_args(name="beat", beat=(0,0,1), count =1)
    def on_beat(beat: int):
        if(beat==1):
            beat_counter[None] += 1  # Increment beat count
            if(beat_started[None]==0):
                beat_started[None]=1
            last_beat_frame[None] = frame_counter[None]
        

    @tv.render
    def _():
        tv.px.diffuse(0.99)
        tv.v.move(tv.p, 10.0)
        frame_counter[None] += 1

    # Reset beat state if no beat detected recently
        if frame_counter[None] - last_beat_frame[None] > 120:
            beat_started[None] = 0
        if(beat_started[None]==0):
            tv.v.attract(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)  
        else:

            # Decide behavior based on beat count (even = repel, odd = attract)
            if beat_counter[None] % 2 == 0:
                tv.v.repel(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)

            else:
                tv.v.attract(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)

        tv.px.particles(tv.p, tv.s.species())
        return tv.px

if __name__ == '__main__':
    run(main)
