import taichi as ti
from tolvera import Tolvera, run



def main(**kwargs):
    tv = Tolvera(**kwargs, species=5, particles=1000)

    # Fields to store beat state
    beat_counter = ti.field(dtype=ti.i32, shape=())  
    beat_started = ti.field(dtype=ti.i32, shape=())  # acts like a boolean
    last_beat_frame = ti.field(dtype=ti.i32, shape=())
    frame_counter = ti.field(dtype=ti.i32, shape=())
    beat_started[None] = 0

    @ti.kernel
    def reset_particles():
        for i in range(tv.p.n):
            tv.p[i].pos.x = tv.x / 2
            tv.p[i].pos.y = tv.y / 2
            tv.p[i].vel.x = 0
            tv.p[i].vel.y = 0

    # reset_particles()


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
        # print(beat_started[None])
        frame_counter[None] += 1

    # Reset beat state if no beat detected recently
        if frame_counter[None] - last_beat_frame[None] > 120:  # e.g., 60 frames = 1 second
            beat_started[None] = 0
        if(beat_started[None]==0):
            tv.v.attract(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)  
        else:

            # Decide behavior based on beat count (even = repel, odd = attract)
            if beat_counter[None] % 2 == 0:
                # print("repel")
                tv.v.repel(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)
                # tv.v.flock(tv.p)
                # tv.v.attract(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)

            else:
                # print("attract")
                # tv.v.swarm(tv.p)
                tv.v.attract(tv.p, [tv.x/2, tv.y/2], 500000000.0, tv.x)

        tv.px.particles(tv.p, tv.s.species())
        return tv.px

if __name__ == '__main__':
    run(main)
