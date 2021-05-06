---
classes: wide
title: "Markup: Post with Code/Images"
excerpt: "Displaying code and images in posts."
header:
  teaser: "http://farm9.staticflickr.com/8426/7758832526_cc8f681e48_c.jpg"
---


If you want to display two or three images next to each other responsively use `figure` with the appropriate `class`.  
Each instance of `figure` is auto-numbered and displayed in the caption.


### Figures (for images or video)

#### One Figure

<figure>
	<a href="http://farm9.staticflickr.com/8426/7758832526_cc8f681e48_b.jpg"><img src="http://farm9.staticflickr.com/8426/7758832526_cc8f681e48_c.jpg"></a>
	<figcaption><a href="http://www.flickr.com/photos/80901381@N04/7758832526/">Morning Fog Emerging From Trees by A Guy Taking Pictures, on Flickr</a>.</figcaption>
</figure>


#### Two Up

Apply the `half` class like so to display two images side by side that share the same caption.

```html
<figure class="half">
    <a href="/assets/images/image-filename-1-large.jpg"><img src="/assets/images/image-filename-1.jpg"></a>
    <a href="/assets/images/image-filename-2-large.jpg"><img src="/assets/images/image-filename-2.jpg"></a>
    <figcaption>Caption describing these two images.</figcaption>
</figure>
```

And you'll get something that looks like this:

<figure class="half">
	<a href="/assets/images/star_galaxy_1200x777.jpg"><img src="/assets/images/star_galaxy_1200x777.jpg"></a>
	<img src="/assets/images/star_galaxy_1200x777.jpg"></a>
	<figcaption>Caption for both images: left has link, right does not.</figcaption>
</figure>

