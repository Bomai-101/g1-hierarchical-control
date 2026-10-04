"""Task-layer contract tests, including expired, invalid and reordered messages."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from g1_control.hierarchy import TaskRequest,RobotSample,Stage,TaskSequencer

class SequenceTests(unittest.TestCase):
 def new(self,**kwargs):
  return TaskSequencer('demo',[Stage('walk','forward_distance',1.,(.5,0.,0.)),Stage('turn','relative_heading',.4,(.5,0.,.2))],**kwargs)
 def sample(self,t,x=0.,y=0.,heading=0.):return RobotSample(t,x,y,heading)
 def req(self,seq,t,task='demo'):return TaskRequest(task,seq,t)
 def test_measured_goals_not_elapsed_schedule(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));d=s.update(self.sample(.1,.9),self.req(1,.1));self.assertEqual(d.stage_name,'walk');d=s.update(self.sample(.2,1.),self.req(2,.2));self.assertEqual(d.stage_name,'turn');d=s.update(self.sample(.3,1.,heading=.4),self.req(3,.3));self.assertEqual(d.task_state,'completed')
 def test_projection_and_relative_heading(self):
  s=self.new();s.update(self.sample(0,heading=1.5707963267948966),self.req(0,0));d=s.update(self.sample(.1,1.,0.,1.5707963267948966),self.req(1,.1));self.assertAlmostEqual(d.progress,0);d=s.update(self.sample(.2,1.,1.,1.5707963267948966),self.req(2,.2));self.assertEqual(d.stage_name,'turn')
 def test_stale_blocks_new_stage_but_holds_command(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));d=s.update(self.sample(.32,1.1));self.assertEqual(d.watchdog_state,'stale');self.assertEqual(d.stage_name,'walk');self.assertEqual(d.command,(.5,0.,0.));self.assertFalse(d.fallback_authorized)
 def test_recovery_needs_two_new_messages(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));s.update(self.sample(.32,1.1));d=s.update(self.sample(.4,1.1),self.req(1,.4));self.assertEqual(d.watchdog_state,'recovering');d=s.update(self.sample(.42,1.1),self.req(1,.4));self.assertEqual(d.watchdog_state,'recovering');d=s.update(self.sample(.5,1.1),self.req(2,.5));self.assertEqual(d.watchdog_state,'recovering');d=s.update(self.sample(.6,1.1),self.req(3,.6));self.assertEqual(d.watchdog_state,'fresh');self.assertEqual(d.stage_name,'turn')
 def test_future_packet_cannot_refresh(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));d=s.update(self.sample(.32),self.req(1,1));self.assertEqual(d.watchdog_state,'stale');self.assertEqual(s.events[-1]['event'],'message_rejected')
 def test_expired_packet_cannot_refresh(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));d=s.update(self.sample(.4),self.req(1,0));self.assertEqual(d.watchdog_state,'stale')
 def test_wrong_task_cannot_start(self):
  s=self.new();d=s.update(self.sample(0),self.req(0,0,'other'));self.assertEqual(d.task_state,'waiting');self.assertIsNone(d.command)
 def test_new_sequence_with_older_timestamp_is_rejected(self):
  s=self.new();s.update(self.sample(.1),self.req(2,.1));s.update(self.sample(.2),self.req(3,.05));self.assertEqual(s.last_request.sequence,2)
 def test_boundary_age_and_abort(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));self.assertEqual(s.update(self.sample(.3)).watchdog_state,'fresh');s.update(self.sample(.32));d=s.update(self.sample(2.32));self.assertEqual(d.task_state,'aborted');self.assertEqual(d.reason,'communication_timeout')
 def test_missing_start_message_aborts(self):
  s=self.new();s.update(self.sample(0));d=s.update(self.sample(2));self.assertEqual(d.task_state,'aborted')
 def test_stage_timeout(self):
  s=self.new();s.update(self.sample(0),self.req(0,0));d=s.update(self.sample(8),self.req(1,8));self.assertEqual(d.reason,'stage_timeout')
 def test_time_reversal_and_terminal_reuse_rejected(self):
  s=self.new();s.update(self.sample(0),self.req(0,0))
  with self.assertRaises(ValueError):s.update(self.sample(0))
  s.update(self.sample(8),self.req(1,8))
  with self.assertRaises(RuntimeError):s.update(self.sample(8.1),self.req(2,8.1))
 def test_negative_heading_goal(self):
  s=TaskSequencer('demo',[Stage('right','relative_heading',-.4,(.5,0.,-.2))]);s.update(self.sample(0,heading=3.2),self.req(0,0));d=s.update(self.sample(.1,heading=2.8),self.req(1,.1));self.assertEqual(d.task_state,'completed')
 def test_logging_toggle_has_same_decisions(self):
  a=self.new(record_events=True);b=self.new(record_events=False)
  for t,x,seq in [(0,0,0),(.32,1.1,None),(.4,1.1,1),(.5,1.1,2)]:
   request=self.req(seq,t) if seq is not None else None;self.assertEqual(a.update(self.sample(t,x),request),b.update(self.sample(t,x),request))
  self.assertTrue(a.events);self.assertFalse(b.events)

 def test_turn_goal_waits_for_confirmed_handoff_with_extra_rotation(self):
  s=TaskSequencer('demo',[Stage('turn','relative_heading',.4,(.5,0.,.2)),Stage('walk','forward_distance',1.,(.5,0.,0.))])
  s.update(self.sample(0),self.req(0,0))
  held=s.update(self.sample(.32,heading=.41))
  self.assertEqual(held.stage_name,'turn');self.assertAlmostEqual(held.progress,.41)
  first=s.update(self.sample(.4,heading=.45),self.req(1,.4))
  self.assertEqual(first.watchdog_state,'recovering');self.assertEqual(first.command,(.5,0.,.2))
  resumed=s.update(self.sample(.5,x=.2,heading=.48),self.req(2,.5))
  self.assertEqual(resumed.stage_name,'walk');self.assertEqual(resumed.command,(.5,0.,0.));self.assertEqual(resumed.progress,0.)
  completed=next(e for e in s.events if e['event']=='stage_complete')
  self.assertEqual(completed['time_s'],.5);self.assertAlmostEqual(completed['progress'],.48)

 def test_skill_command_contract(self):
  from g1_control.hierarchy.skill_interface import VelocityCommand
  self.assertEqual(VelocityCommand(.5,0.,.2).values,(.5,0.,.2))
  for args in [(float('nan'),0,0),(1.1,0,0),(.5,0,1.1)]:
   with self.assertRaises(ValueError):VelocityCommand(*args)
 def test_stage_invalid_goal_or_timeout(self):
  for goal,timeout in [(0.,8.),(1.,0.),(float('nan'),8.)]:
   with self.assertRaises(ValueError):Stage('bad','forward_distance',goal,(.5,0,0),timeout)

if __name__=='__main__':unittest.main()
