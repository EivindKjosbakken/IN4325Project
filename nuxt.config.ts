// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
	devServer: {
		proxy: {
		  '^/api': {
			target: 'http://localhost:3000',
			changeOrigin: true
		  },
		}
	  }
})
