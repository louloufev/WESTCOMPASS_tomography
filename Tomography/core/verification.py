#supposed to launch this script after having loaded an inversion results

Inversion_results

#plot results
Inversion_results.denoising()
Inversion_results.plot_bob(197)
plt.savefig('native.png')
plt.close()
#create a video

Inversion_results.create_video(percentile_inf = 0, percentile_sup =100)

#change inversion method
Inversion_results.ParamsVid.inversion_parameter['min_visibility_node'] = 0
Inversion_results.redo_inversion_results() #recalculate inversion

Inversion_results.denoising() #recalculate denoising
Inversion_results.plot_bob(197)
plt.savefig('new inversion method.png')
plt.close()

#change video parameters

Inversion_results.ParamsVid.dict_vid['sigma'] = 4
Inversion_results.redo_video()
Inversion_results.redo_inversion_results() #recalculate inversion

Inversion_results.denoising() #recalculate denoising
Inversion_results.plot_bob(197)

plt.savefig('higher gaussian filter.png')
plt.close()