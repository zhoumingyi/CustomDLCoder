/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <sys/time.h>
// add files
#include "zbkxmb.cc"
#include "tkpivl.cc"
#include "pfkppy.cc"
#include "xygmrc.cc"
#include "gscfvz.cc"
#include "uaxlvy.cc"
#include "bfdzsv.cc"
#include "avjdet.cc"
#include "perwxz.cc"
#include "ozsgld.cc"
#include "ilxrse.cc"
#include "qvyngp.cc"
#include "updrhs.cc"
#include "okatba.cc"
#include "pmjcml.cc"
#include "ybxkql.cc"
#include "sztaex.cc"
#include "ofkyef.cc"
#include "wjpcjx.cc"
#include "qrrkmq.cc"
#include "fcitef.cc"
#include "bwugmg.cc"
#include "xczkro.cc"
#include "sdqolk.cc"
#include "pqjycp.cc"
#include "xeaeqt.cc"
#include "btvzxy.cc"
#include "nvufml.cc"
#include "eszurx.cc"
#include "rpxxbq.cc"
#include "xeuwux.cc"
// end files


extern "C" const float* coder(float *input_v) {
//  timeval t_start, t_end; 
//  gettimeofday( &t_start, NULL);
 const Eigen::ThreadPoolDevice* device = tflite::eigen_support::CreateThreadPoolDevice(-1);
 auto* cpu_backend = new CpuBackendContext();
 cpu_backend->SetMaxNumThreads(-1);
  // add function
 auto* out_89 = input_v;
 auto* out_31 = zbkxmb::zbkxmb(out_89, device);
 auto* out_49 = tkpivl::tkpivl(out_31, cpu_backend);
 auto* out_51 = pfkppy::pfkppy(out_49, device);
 auto* out_53 = xygmrc::xygmrc(out_51, cpu_backend);
 auto* out_55 = gscfvz::gscfvz(out_53, device);
 auto* out_57 = uaxlvy::uaxlvy(out_55, cpu_backend);
 auto* out_59 = bfdzsv::bfdzsv(out_57, device);
 auto* out_61 = avjdet::avjdet(out_59, cpu_backend);
 auto* out_63 = perwxz::perwxz(out_61, device);
 auto* out_65 = ozsgld::ozsgld(out_63, cpu_backend);
 auto* out_67 = ilxrse::ilxrse(out_65, device);
 auto* out_69 = qvyngp::qvyngp(out_67, cpu_backend);
 auto* out_71 = updrhs::updrhs(out_69, device);
 auto* out_73 = okatba::okatba(out_71, cpu_backend);
 auto* out_75 = pmjcml::pmjcml(out_73, device);
 auto* out_77 = ybxkql::ybxkql(out_75, cpu_backend);
 auto* out_79 = sztaex::sztaex(out_77, device);
 auto* out_81 = ofkyef::ofkyef(out_79, cpu_backend);
 auto* out_83 = wjpcjx::wjpcjx(out_81, device);
 auto* out_33 = qrrkmq::qrrkmq(out_83, cpu_backend);
 auto* out_35 = fcitef::fcitef(out_33, device);
 auto* out_37 = bwugmg::bwugmg(out_35, cpu_backend);
 auto* out_39 = xczkro::xczkro(out_37, device);
 auto* out_41 = sdqolk::sdqolk(out_39, cpu_backend);
 auto* out_43 = pqjycp::pqjycp(out_41, device);
 auto* out_45 = xeaeqt::xeaeqt(out_43, cpu_backend);
 auto* out_47 = btvzxy::btvzxy(out_45, device);
 auto* out_27 = nvufml::nvufml(out_47);
 auto* out_28 = eszurx::eszurx(out_27, device);
 auto* out_87 = rpxxbq::rpxxbq(out_28, cpu_backend);
 auto* out_85 = xeuwux::xeuwux(out_87, cpu_backend);
 return out_85;
  // end function
}
